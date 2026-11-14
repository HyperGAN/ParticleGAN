# Forge experiments by tier

Current task assignments, grouped by goal view and qualification tier. Required tasks gate progression; ranking and diagnostic tasks retain their declared roles.

Catalog: **198 tasks**; **160 assigned** to at least one view; **38 unassigned**. Showing **32/32 views**.

Declared priors across the catalog: **88 MoGParticlePrior**, **110 ParticlePrior** (including **10 nonsampled parameter controls**). Every experiment defines `execution.prior` explicitly; candidate and API defaults cannot supply it. `kind: mog` selects `MoGParticlePrior`; `kind: particle_cloud` selects `ParticlePrior`. Sigma alone does not identify the code path. Ordinary Forge MoG tasks require positive sigma; archived zero-sigma MoG evidence keeps its recorded kind. Task sigma is absolute; API demonstrations may instead record the recipe's relative `sigma_rel`.

Reproducible comparisons use the fixed screening seed `0` and candidate-independent named RNG streams. Within each task, candidates share architecture, data law, batch size, prior, initialization, training budget, evaluation cadence and sampling law. Only the declared trainer change varies. The initialization column exposes fixed controls and component policies that take precedence over the deterministic orthogonal fallback; these are separate comparison cohorts. Historical results retain their original bindings.

Regenerate from the repository root with `python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md`. Add `--json` for machine-readable output (use a `.json` output path when saving). Regeneration reads declarations and published artifacts and launches no training.

Tier 1 is smoke, Tier 2 is quality, Tier 3 is endurance, and Tier 4 is stress_and_formulations. Views may leave later tiers empty. Placement follows each view's policy.

Held tasks are still run and shown but are not counted until one recipe passes them together with every counted task. A diagnostic task removed from the count is carried by another counted task or cannot run as declared. Each such role links to its explanation; a view whose highest tier has only held tasks never shows PASS: a candidate that passes every counted tier gets HELD, and a counted failure still shows as a failure.

Steps and timeouts are declared per task, rather than measured costs. Tasks in an uninterrupted execution group share one run; their budgets must not be added together. Continuation rows distinguish total steps from additional or extension steps.

## Review experiments and solutions

Use the tier tables to see the current requirements, then open each task's experiment guide for its question, numerical gates, recorded configurations and related training GIFs.

- [Current solution leaderboard](technique-inventory.md): selected configurations and complete Forge tier denominators.
- [Research memory](EXPERIMENT_MEMORY.md): hypotheses, comparisons, failure explanations and recommendations.
- [All retained public-API questions](../toy_audit/api_contract/QUESTION_RANKING.md): the wider research catalog, explanations, results and goal GIFs.
- [Public-API experiment readout](../toy_audit/api_contract/RUN_REPORT.md): completed protocols and failed numerical bounds.
- [Public-API test contract](../toy_audit/api_contract/README.md): reproduction commands and exact scope of the demos.

Recorded Forge outcomes below are lookups by exact task ID from the current solution publication; their source revision and runtime remain explicit. Related API variants illustrate the question under their own gates, recipes, priors, initialization, budgets and sampling laws. Their PASS results do not qualify a different Forge task or the latest checkout. The solution leaderboard remains the single ranking for its goal.

This report follows changing declarations and published evidence; it selects no release winner. Release selection requires full Forge qualification and review of the winner. A smoke-qualified entry or historical pass alone does not establish release readiness.

## Current tier assignments

| View | Revision | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Declared calibration |
| --- | ---: | --- | --- | --- | --- | --- |
| [adaptation](../../configs/forge/views/adaptation.json) | 4 | 3 required | 12 required, 7 held | 1 required | 0 tasks | provisional |
| [bcap-develop-integration-deeper-diagnostic-v1](../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json) | 1 | 21 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [bcap-projection-baseline-repair-diagnostic-v1](../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json) | 1 | 16 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [bcap-tier1-stability-diagnostic-v1](../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json) | 1 | 1 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [bcap-tier1-stability-repairs-diagnostic-v1](../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json) | 1 | 1 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [bcap_convolution_images](../../configs/forge/views/bcap_convolution_images.json) | 1 | 4 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [bispf_fastpath_quality](../../configs/forge/views/bispf_fastpath_quality.json) | 1 | 2 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [bispf_throughput_endurance](../../configs/forge/views/bispf_throughput_endurance.json) | 2 | 3 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [bispf_throughput_small](../../configs/forge/views/bispf_throughput_small.json) | 2 | 2 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json) | 5 | 4 required | 12 required, 7 held | 6 required, 1 diagnostic (1 removed from the count) | 0 tasks | provisional |
| [dcgan_performance](../../configs/forge/views/dcgan_performance.json) | 1 | 6 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [dcgan_performance_cpu_smoke](../../configs/forge/views/dcgan_performance_cpu_smoke.json) | 1 | 1 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [ddgan-conditional-repair-v1](../../configs/forge/views/ddgan-conditional-repair-v1.json) | 2 | 2 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [discriminator_stability](../../configs/forge/views/discriminator_stability.json) | 17 | 11 required, 3 stretch, 2 diagnostic | 18 required, 2 stretch, 7 held | 1 required, 1 held, 1 diagnostic (1 removed from the count) | 10 stretch, 5 held | provisional |
| [formulation_comparison](../../configs/forge/views/formulation_comparison.json) | 3 | 3 required | 12 required, 7 held, 15 diagnostic | 1 required, 1 diagnostic (1 removed from the count) | 0 tasks | provisional |
| [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json) | 6 | 3 required | 12 required, 7 held, 13 diagnostic | 1 required, 1 diagnostic (1 removed from the count) | 0 tasks | provisional |
| [k3p_two_pole_horizon](../../configs/forge/views/k3p_two_pole_horizon.json) | 1 | 2 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [lazy_cadence_coverage](../../configs/forge/views/lazy_cadence_coverage.json) | 1 | 30 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [misgan-bispf-research-v1](../../configs/forge/views/misgan-bispf-research-v1.json) | 2 | 1 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [particle_interpolation_stretch](../../configs/forge/views/particle_interpolation_stretch.json) | 1 | 3 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [policy_cloud_coverage](../../configs/forge/views/policy_cloud_coverage.json) | 1 | 7 required | 21 required | 7 required | 0 tasks | undeclared |
| [policy_cloud_coverage_as_served](../../configs/forge/views/policy_cloud_coverage_as_served.json) | 1 | 7 required | 21 required | 7 required | 0 tasks | undeclared |
| [prior-ablation-full-suite-v1](../../configs/forge/views/prior-ablation-full-suite-v1.json) | 1 | 48 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [quality_coverage](../../configs/forge/views/quality_coverage.json) | 4 | 3 required | 12 required, 7 held | 0 tasks | 0 tasks | provisional |
| [r1r2-repair-word-retention-20261022](../../configs/forge/views/r1r2-repair-word-retention-20261022.json) | 1 | 2 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [recipe-prior-refactor-k3p-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json) | 1 | 23 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [recipe-prior-refactor-ka2-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json) | 1 | 23 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [recipe-prior-refactor-r1r2-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json) | 1 | 23 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [recipe-prior-refactor-release07-gan-v3-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json) | 1 | 23 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |
| [selected-family-coverage-diagnostic-v1](../../configs/forge/views/selected-family-coverage-diagnostic-v1.json) | 1 | 1 diagnostic | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [tier1_policy_coverage](../../configs/forge/views/tier1_policy_coverage.json) | 1 | 7 required | 0 tasks | 0 tasks | 0 tasks | undeclared |
| [vaegan_coverage](../../configs/forge/views/vaegan_coverage.json) | 1 | 2 diagnostic | 0 tasks | 0 tasks | 0 tasks | provisional |

## adaptation

Declaration: [adaptation](../../configs/forge/views/adaptation.json); revision 4; goal: `adaptation`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/adaptation.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

12 required, 7 held.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_mass) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_width) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_anisotropic) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_overlap) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_bars4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_blobs4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_intensity2) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

1 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bcap-develop-integration-deeper-diagnostic-v1

Declaration: [bcap-develop-integration-deeper-diagnostic-v1](../../configs/forge/views/bcap-develop-integration-deeper-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

21 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bcap-projection-baseline-repair-diagnostic-v1

Declaration: [bcap-projection-baseline-repair-diagnostic-v1](../../configs/forge/views/bcap-projection-baseline-repair-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

16 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bcap-tier1-stability-diagnostic-v1

Declaration: [bcap-tier1-stability-diagnostic-v1](../../configs/forge/views/bcap-tier1-stability-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

1 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bcap-tier1-stability-repairs-diagnostic-v1

Declaration: [bcap-tier1-stability-repairs-diagnostic-v1](../../configs/forge/views/bcap-tier1-stability-repairs-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision8 acquisition/hold separation is provisional and requires bounded calibration before default adoption. Historical task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

1 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bcap_convolution_images

Declaration: [bcap_convolution_images](../../configs/forge/views/bcap_convolution_images.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Four source-bound image diagnostics do not qualify a new source or adopt public defaults.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

4 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bispf_fastpath_quality

Declaration: [bispf_fastpath_quality](../../configs/forge/views/bispf_fastpath_quality.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [img_stripes2_w64_optimizer_diagnostic](../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 300 | — |
| [img_bars4_w64_optimizer_diagnostic](../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bispf_throughput_endurance

Declaration: [bispf_throughput_endurance](../../configs/forge/views/bispf_throughput_endurance.json); revision 2; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

3 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## bispf_throughput_small

Declaration: [bispf_throughput_small](../../configs/forge/views/bispf_throughput_small.json); revision 2; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## clockfree_continuous

Declaration: [clockfree_continuous](../../configs/forge/views/clockfree_continuous.json); revision 5; goal: `clockfree_continuous`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/clockfree_continuous.md).

### Tier 1: smoke

4 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

12 required, 7 held.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_mass) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_width) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_anisotropic) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_overlap) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_bars4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_blobs4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_intensity2) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

6 required, 1 diagnostic (1 removed from the count).

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | [removed: phase of ring_extension (counted there)](test-dispositions-20261015/README.md#ring_hold) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [grid100_14k](../../configs/forge/tasks/grid100_14k.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100](../../configs/forge/tasks/grid100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100](../../configs/forge/tasks/rotated100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100](../../configs/forge/tasks/staggered100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## dcgan_performance

Declaration: [dcgan_performance](../../configs/forge/views/dcgan_performance.json); revision 1; goal: `image_performance`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

6 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [dcgan_performance_r32](../../configs/forge/tasks/dcgan_performance_r32.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-r32) | dcgan_performance_v1 / image_training_performance | 13 | 180 | — |
| [dcgan_performance_r64](../../configs/forge/tasks/dcgan_performance_r64.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-r64) | dcgan_performance_v1 / image_training_performance | 13 | 180 | — |
| [dcgan_performance_r128](../../configs/forge/tasks/dcgan_performance_r128.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-r128) | dcgan_performance_v1 / image_training_performance | 13 | 180 | — |
| [dcgan_performance_r256](../../configs/forge/tasks/dcgan_performance_r256.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-r256) | dcgan_performance_v1 / image_training_performance | 13 | 300 | — |
| [dcgan_performance_b64](../../configs/forge/tasks/dcgan_performance_b64.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-b64) | dcgan_performance_v1 / image_training_performance | 13 | 180 | — |
| [dcgan_performance_w128](../../configs/forge/tasks/dcgan_performance_w128.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-w128) | dcgan_performance_v1 / image_training_performance | 13 | 180 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## dcgan_performance_cpu_smoke

Declaration: [dcgan_performance_cpu_smoke](../../configs/forge/views/dcgan_performance_cpu_smoke.json); revision 1; goal: `image_performance`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

1 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [dcgan_performance_cpu_smoke](../../configs/forge/tasks/dcgan_performance_cpu_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; — | [Question, results, GIFs](#experiment-dcgan-performance-cpu-smoke) | dcgan_performance_v1 / image_training_performance | 3 | 120 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## ddgan-conditional-repair-v1

Declaration: [ddgan-conditional-repair-v1](../../configs/forge/views/ddgan-conditional-repair-v1.json); revision 2; goal: `ddgan_conditional_repair`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ddgan_repair_gaussian_chain_v1](../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json) | diagnostic | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-gaussian-chain) | public_api_model_toy / transfer_sustained | 2500 | 600 | — |
| [ddgan_repair_bimodal_chain_wall_v2](../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json) | diagnostic | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-bimodal-chain) | public_api_model_toy / transfer_sustained | 5000 | 1200 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## discriminator_stability

Declaration: [discriminator_stability](../../configs/forge/views/discriminator_stability.json); revision 17; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

Tier 4, the highest tier, has only held tasks, so while they are held this view never shows PASS: a candidate that passes every counted tier gets HELD.

### Tier 1: smoke

11 required, 3 stretch, 2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [zero_tanh_smoke](../../configs/forge/tasks/zero_tanh_smoke.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-zero-tanh-smoke) | transfer_vector / zero_tanh_smoke | 1000 | 120 | — |
| [model_ae_gan_multimodal](../../configs/forge/tasks/model_ae_gan_multimodal.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-model-ae-gan-multimodal) | public_api_model_toy / transfer_sustained | 1200 | 600 | — |
| [model_ae_reconstruction](../../configs/forge/tasks/model_ae_reconstruction.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-model-ae-reconstruction) | public_api_model_toy / transfer_sustained | 1200 | 600 | — |
| [model_vae_categorical](../../configs/forge/tasks/model_vae_categorical.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-api-model-vae-categorical) | public_api_model_toy / transfer_sustained | 1200 | 300 | — |
| [model_vae_hard](../../configs/forge/tasks/model_vae_hard.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-api-model-vae-hard) | public_api_model_toy / transfer_sustained | 1200 | 300 | — |
| [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [precision_gaussian_fp32](../../configs/forge/tasks/precision_gaussian_fp32.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-precision-gaussian) | transfer_vector / transfer_sustained | 1000 | 300 | — |
| [precision_gaussian_mxfp8](../../configs/forge/tasks/precision_gaussian_mxfp8.json) | [stretch: frontier stretch benchmark beyond the required suite](precision-stretch/README.md#precision-stretch) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-precision-gaussian) | transfer_vector / transfer_sustained | 1000 | 300 | — |
| [precision_gaussian_nvfp4](../../configs/forge/tasks/precision_gaussian_nvfp4.json) | [stretch: frontier stretch benchmark beyond the required suite](precision-stretch/README.md#precision-stretch) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-precision-gaussian) | transfer_vector / transfer_sustained | 1000 | 300 | — |
| [interp_sparse_arc](../../configs/forge/tasks/interp_sparse_arc.json) | [stretch: frontier stretch benchmark beyond the required suite](particle-interpolation-stretch/README.md#interp_sparse_arc) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-interp-sparse-arc) | particle_interpolation / interp_sparse_arc | 600 | 300 | — |

### Tier 2: quality

18 required, 2 stretch, 7 held.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | required | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [model_ae_ddgan_gaussian_chain](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json) | required | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ae-ddgan-gaussian-chain) | public_api_model_toy / transfer_sustained | 2500 | 180 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_mass) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [model_ddgan_gaussian_chain](../../configs/forge/tasks/model_ddgan_gaussian_chain.json) | required | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-gaussian-chain) | public_api_model_toy / transfer_sustained | 2500 | 600 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_width) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_anisotropic) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_overlap) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_bars4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_blobs4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_intensity2) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [model_ae_ddgan_bimodal_chain](../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json) | required | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ae-ddgan-bimodal-chain) | public_api_model_toy / transfer_sustained | 5000 | 300 | [model_ae_ddgan_gaussian_chain](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json) (gate) |
| [model_ddgan_bimodal_chain](../../configs/forge/tasks/model_ddgan_bimodal_chain.json) | required | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-bimodal-chain) | public_api_model_toy / transfer_sustained | 5000 | 600 | [model_ddgan_gaussian_chain](../../configs/forge/tasks/model_ddgan_gaussian_chain.json) (gate) |
| [interp_img_patch_shift](../../configs/forge/tasks/interp_img_patch_shift.json) | [stretch: frontier stretch benchmark beyond the required suite](particle-interpolation-stretch/README.md#interp_img_patch_shift) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-interp-img-patch-shift) | particle_interpolation / interp_img_patch_shift | 800 | 300 | — |
| [interp_saddle_barycentric](../../configs/forge/tasks/interp_saddle_barycentric.json) | [stretch: frontier stretch benchmark beyond the required suite](particle-interpolation-stretch/README.md#interp_saddle_barycentric) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-interp-saddle-barycentric) | particle_interpolation / interp_saddle_barycentric | 800 | 300 | — |

### Tier 3: endurance

1 required, 1 held, 1 diagnostic (1 removed from the count).

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | [removed: phase of ring_extension (counted there)](test-dispositions-20261015/README.md#ring_hold) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [zero_tanh_rail_escape](../../configs/forge/tasks/zero_tanh_rail_escape.json) | [held: no recipe has passed its current declaration](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-zero-tanh-rail-escape) | transfer_vector / zero_tanh_rail_escape | 2000 | 300 | [zero_tanh_smoke](../../configs/forge/tasks/zero_tanh_smoke.json) (gate) |

### Tier 4: stress_and_formulations

10 stretch, 5 held. Nothing here is counted while these tasks are held, so this tier's status is HELD: it neither passes nor stops later tiers.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [grid100_mlp16](../../configs/forge/tasks/grid100_mlp16.json) | [held: no recipe has passed its current declaration](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | [grid100](../../configs/forge/tasks/grid100.json) (gate) |
| [grid100_res16_tanh](../../configs/forge/tasks/grid100_res16_tanh.json) | [held: no recipe has passed its current declaration](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | [grid100](../../configs/forge/tasks/grid100.json) (gate) |
| [misgan_block_recovery](../../configs/forge/tasks/misgan_block_recovery.json) | [held: no recipe has passed its current declaration](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-misgan-block-recovery) | transfer_vector / misgan_block_recovery | 7000 | 1800 | — |
| [misgan_block_neural_imputer](../../configs/forge/tasks/misgan_block_neural_imputer.json) | [held: no recipe has passed its current declaration](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-misgan-block-neural-imputer) | transfer_vector / misgan_block_neural_imputer | 7000 | 1800 | — |
| [ring16_bigan_continuous](../../configs/forge/tasks/ring16_bigan_continuous.json) | [held: no recipe has passed its current declaration](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-bigan-continuous) | transfer_vector / ring16_bigan_continuous | 4000 | 600 | — |
| [misgan_mcar_p50_recovery](../../configs/forge/tasks/misgan_mcar_p50_recovery.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-misgan-mcar-p50-recovery) | transfer_vector / misgan_mcar_p50_recovery | 4000 | 1800 | — |
| [misgan_mcar_p80_anchored](../../configs/forge/tasks/misgan_mcar_p80_anchored.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-misgan-mcar-p80-anchored) | transfer_vector / misgan_mcar_p80_anchored | 4000 | 1800 | — |
| [pacgan_unequal_mass](../../configs/forge/tasks/pacgan_unequal_mass.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.05) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-pacgan-unequal-mass) | transfer_vector / pacgan_unequal_mass | 3000 | 600 | — |
| [pacgan8_grid100](../../configs/forge/tasks/pacgan8_grid100.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.05) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-pacgan8-grid100) | transfer_vector / pacgan8_grid100 | 4000 | 600 | — |
| [alice_spiral_bigan](../../configs/forge/tasks/alice_spiral_bigan.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.06) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-alice-spiral-bigan) | transfer_vector / alice_spiral_bigan | 4000 | 600 | — |
| [infogan_anisotropic_modes](../../configs/forge/tasks/infogan_anisotropic_modes.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.05) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-infogan-anisotropic-modes) | transfer_vector / infogan_anisotropic_modes | 3500 | 600 | — |
| [cgan_overlapping_classes](../../configs/forge/tasks/cgan_overlapping_classes.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cgan-overlapping-classes) | transfer_vector / cgan_overlapping_classes | 3000 | 600 | — |
| [cramer_energy_overlap](../../configs/forge/tasks/cramer_energy_overlap.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cramer-energy-overlap) | transfer_vector / cramer_energy_overlap | 3000 | 600 | — |
| [cyclegan_unpaired_constellation](../../configs/forge/tasks/cyclegan_unpaired_constellation.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.08) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cyclegan-unpaired-constellation) | transfer_vector / cyclegan_unpaired_constellation | 3500 | 600 | — |
| [grid100_res16_scaled_tanh](../../configs/forge/tasks/grid100_res16_scaled_tanh.json) | [stretch: frontier stretch benchmark beyond the required suite](test-dispositions-20261015/README.md#unresolved) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | [grid100](../../configs/forge/tasks/grid100.json) (gate) |

## formulation_comparison

Declaration: [formulation_comparison](../../configs/forge/views/formulation_comparison.json); revision 3; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/formulation_comparison.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

12 required, 7 held, 15 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_mass) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_width) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_anisotropic) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_overlap) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_bars4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_blobs4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_intensity2) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

1 required, 1 diagnostic (1 removed from the count).

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | [removed: phase of ring_extension (counted there)](test-dispositions-20261015/README.md#ring_hold) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## host_profile_transfer

Declaration: [host_profile_transfer](../../configs/forge/views/host_profile_transfer.json); revision 6; goal: `host_profile_transfer`.

Declared calibration status: **provisional**.

Host-profile transfer and full current-cohort positive/negative reference calibration remain required; no archived pass is imported.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/host_profile_transfer.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

12 required, 7 held, 13 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_mass) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_width) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_anisotropic) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_overlap) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_bars4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_blobs4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_intensity2) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

1 required, 1 diagnostic (1 removed from the count).

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | [removed: phase of ring_extension (counted there)](test-dispositions-20261015/README.md#ring_hold) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## k3p_two_pole_horizon

Declaration: [k3p_two_pole_horizon](../../configs/forge/views/k3p_two_pole_horizon.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Bounded budget/schedule diagnostic supplies no ordinary qualification, screen calibration or default adoption.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole_800_schedule80_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 800 | 300 | — |
| [two_pole_800_schedule800_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 800 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## lazy_cadence_coverage

Declaration: [lazy_cadence_coverage](../../configs/forge/views/lazy_cadence_coverage.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

30 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## misgan-bispf-research-v1

Declaration: [misgan-bispf-research-v1](../../configs/forge/views/misgan-bispf-research-v1.json); revision 2; goal: `misgan_recovery_and_imputation`.

Declared calibration status: **provisional**.

Scoped research diagnostic; no ordinary qualification or default adoption.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

1 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [model_misgan_four_mode](../../configs/forge/tasks/model_misgan_four_mode.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-model-misgan-four-mode) | public_api_model_toy / transfer_sustained | 1200 | 1800 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## particle_interpolation_stretch

Declaration: [particle_interpolation_stretch](../../configs/forge/views/particle_interpolation_stretch.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Three zero-sigma particle interpolation stretch toys do not establish whole-suite screen calibration or default adoption.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

3 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp_sparse_arc](../../configs/forge/tasks/interp_sparse_arc.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-interp-sparse-arc) | particle_interpolation / interp_sparse_arc | 600 | 300 | — |
| [interp_img_patch_shift](../../configs/forge/tasks/interp_img_patch_shift.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-interp-img-patch-shift) | particle_interpolation / interp_img_patch_shift | 800 | 300 | — |
| [interp_saddle_barycentric](../../configs/forge/tasks/interp_saddle_barycentric.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-interp-saddle-barycentric) | particle_interpolation / interp_saddle_barycentric | 800 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## policy_cloud_coverage

Declaration: [policy_cloud_coverage](../../configs/forge/views/policy_cloud_coverage.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/two_pole_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [gaussian1d_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_smoke_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [ring16_acquisition_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring16_acquisition_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [clockfree_audit_measurement_v1_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_measurement_v1_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1-policy-cloud-coverage-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [unused_token_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unused_token_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ae_gan_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [five_word_joint_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_smoke_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |

### Tier 2: quality

21 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_stability_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_stability_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_smoke_policy_cloud_coverage_v1.json) (checkpoint) |
| [grid100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_stripes2_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_stripes2_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_bars4_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_blobs4_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_intensity2_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [mode_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mode_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_two_broad_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_unequal_mass_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_unequal_width_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_anisotropic_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_overlap_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_spiral_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [five_word_joint_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_smoke_policy_cloud_coverage_v1.json) (checkpoint) |
| [trajectory_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/trajectory_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/residual_student_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/cover_leftover_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [unipolar_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unipolar_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [mid_scale_identity_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mid_scale_identity_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |

### Tier 3: endurance

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-policy-cloud-coverage-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [grid100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_14k_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_policy_cloud_coverage_v1.json) (checkpoint); [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) (gate) |
| [rotated100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_14k_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_policy_cloud_coverage_v1.json) (checkpoint); [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) (gate) |
| [staggered100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_14k_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_policy_cloud_coverage_v1.json) (checkpoint); [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) (gate) |
| [ring_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mode_hold_policy_cloud_coverage_v1.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_extension_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_hold_policy_cloud_coverage_v1.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [target_shift_recovery_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/target_shift_recovery_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mode_hold_policy_cloud_coverage_v1.json) (gate) |

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## policy_cloud_coverage_as_served

Declaration: [policy_cloud_coverage_as_served](../../configs/forge/views/policy_cloud_coverage_as_served.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/two_pole_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [gaussian1d_smoke_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_smoke_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [ring16_acquisition_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring16_acquisition_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [clockfree_audit_measurement_v1_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_measurement_v1_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1-policy-cloud-coverage-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [unused_token_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unused_token_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ae_gan_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [five_word_joint_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_smoke_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |

### Tier 2: quality

21 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_stability_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_stability_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_smoke_policy_cloud_coverage_as_served_v1.json) (checkpoint) |
| [grid100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [img_stripes2_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_stripes2_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_bars4_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_blobs4_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_intensity2_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [mode_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/mode_hold_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_two_broad_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_unequal_mass_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_unequal_width_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_anisotropic_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_overlap_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_spiral_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [five_word_joint_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_hold_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_smoke_policy_cloud_coverage_v1.json) (checkpoint) |
| [trajectory_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/trajectory_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/residual_student_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/cover_leftover_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [unipolar_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unipolar_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [mid_scale_identity_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mid_scale_identity_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |

### Tier 3: endurance

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-policy-cloud-coverage-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [grid100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_14k_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_policy_cloud_coverage_as_served_v1.json) (checkpoint); [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) (gate) |
| [rotated100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_14k_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_policy_cloud_coverage_as_served_v1.json) (checkpoint); [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) (gate) |
| [staggered100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_14k_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_policy_cloud_coverage_as_served_v1.json) (checkpoint); [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json) (gate) |
| [ring_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_hold_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/mode_hold_policy_cloud_coverage_as_served_v1.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_extension_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_hold_policy_cloud_coverage_as_served_v1.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [target_shift_recovery_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/target_shift_recovery_policy_cloud_coverage_as_served_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/mode_hold_policy_cloud_coverage_as_served_v1.json) (gate) |

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## prior-ablation-full-suite-v1

Declaration: [prior-ablation-full-suite-v1](../../configs/forge/views/prior-ablation-full-suite-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

48 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [two_pole](../../configs/forge/tasks/two_pole.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [zero_tanh_smoke](../../configs/forge/tasks/zero_tanh_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-zero-tanh-smoke) | transfer_vector / zero_tanh_smoke | 1000 | 120 | — |
| [model_ae_ddgan_gaussian_chain](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json) | diagnostic | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ae-ddgan-gaussian-chain) | public_api_model_toy / transfer_sustained | 2500 | 180 | — |
| [model_ae_gan_multimodal](../../configs/forge/tasks/model_ae_gan_multimodal.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-model-ae-gan-multimodal) | public_api_model_toy / transfer_sustained | 1200 | 600 | — |
| [model_ae_reconstruction](../../configs/forge/tasks/model_ae_reconstruction.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-model-ae-reconstruction) | public_api_model_toy / transfer_sustained | 1200 | 600 | — |
| [model_ddgan_gaussian_chain](../../configs/forge/tasks/model_ddgan_gaussian_chain.json) | diagnostic | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-gaussian-chain) | public_api_model_toy / transfer_sustained | 2500 | 600 | — |
| [model_vae_categorical](../../configs/forge/tasks/model_vae_categorical.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-api-model-vae-categorical) | public_api_model_toy / transfer_sustained | 1200 | 300 | — |
| [model_vae_hard](../../configs/forge/tasks/model_vae_hard.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-api-model-vae-hard) | public_api_model_toy / transfer_sustained | 1200 | 300 | — |
| [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [model_ae_ddgan_bimodal_chain](../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json) | diagnostic | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ae-ddgan-bimodal-chain) | public_api_model_toy / transfer_sustained | 5000 | 300 | [model_ae_ddgan_gaussian_chain](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json) (gate) |
| [model_ddgan_bimodal_chain](../../configs/forge/tasks/model_ddgan_bimodal_chain.json) | diagnostic | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-bimodal-chain) | public_api_model_toy / transfer_sustained | 5000 | 600 | [model_ddgan_gaussian_chain](../../configs/forge/tasks/model_ddgan_gaussian_chain.json) (gate) |
| [ring_hold](../../configs/forge/tasks/ring_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [ring_extension](../../configs/forge/tasks/ring_extension.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [zero_tanh_rail_escape](../../configs/forge/tasks/zero_tanh_rail_escape.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-zero-tanh-rail-escape) | transfer_vector / zero_tanh_rail_escape | 2000 | 300 | [zero_tanh_smoke](../../configs/forge/tasks/zero_tanh_smoke.json) (gate) |
| [grid100_14k](../../configs/forge/tasks/grid100_14k.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100](../../configs/forge/tasks/grid100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100](../../configs/forge/tasks/rotated100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100](../../configs/forge/tasks/staggered100.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | paired_adaptation / paired_adaptation | 3600 | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate) |
| [misgan_block_recovery](../../configs/forge/tasks/misgan_block_recovery.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-misgan-block-recovery) | transfer_vector / misgan_block_recovery | 7000 | 1800 | — |
| [misgan_block_neural_imputer](../../configs/forge/tasks/misgan_block_neural_imputer.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-misgan-block-neural-imputer) | transfer_vector / misgan_block_neural_imputer | 7000 | 1800 | — |
| [ring16_bigan_continuous](../../configs/forge/tasks/ring16_bigan_continuous.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-bigan-continuous) | transfer_vector / ring16_bigan_continuous | 4000 | 600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## quality_coverage

Declaration: [quality_coverage](../../configs/forge/views/quality_coverage.json); revision 4; goal: `quality_coverage`.

Declared calibration status: **provisional**.

Phase D historical calibration remains required

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/quality_coverage.md).

### Tier 1: smoke

3 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [two_pole](../../configs/forge/tasks/two_pole.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |

### Tier 2: quality

12 required, 7 held.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [trajectory](../../configs/forge/tasks/trajectory.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_mass) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_unequal_width) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_anisotropic) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#vector_overlap) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_bars4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_blobs4) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | [held: no one recipe passes it together with the counted tests](test-dispositions-20261015/README.md#img_intensity2) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | required | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## r1r2-repair-word-retention-20261022

Declaration: [r1r2-repair-word-retention-20261022](../../configs/forge/views/r1r2-repair-word-retention-20261022.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Own-checkpoint word retention only; no ordinary tier qualification, source promotion or global recipe adoption.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## recipe-prior-refactor-k3p-diagnostic-v1

Declaration: [recipe-prior-refactor-k3p-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-k3p-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

23 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## recipe-prior-refactor-ka2-diagnostic-v1

Declaration: [recipe-prior-refactor-ka2-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-ka2-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

23 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## recipe-prior-refactor-r1r2-diagnostic-v1

Declaration: [recipe-prior-refactor-r1r2-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-r1r2-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

23 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## recipe-prior-refactor-release07-gan-v3-diagnostic-v1

Declaration: [recipe-prior-refactor-release07-gan-v3-diagnostic-v1](../../configs/forge/views/recipe-prior-refactor-release07-gan-v3-diagnostic-v1.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **provisional**.

Revision9 recipe-owned prior comparisons are provisional and require bounded calibration before scientific default adoption. Archived task declarations and gates retain their original identities.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

23 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json) | diagnostic | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [trajectory](../../configs/forge/tasks/trajectory.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [residual_student](../../configs/forge/tasks/residual_student.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [unipolar](../../configs/forge/tasks/unipolar.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [cover_leftover](../../configs/forge/tasks/cover_leftover.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json) | diagnostic | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [mode_hold](../../configs/forge/tasks/mode_hold.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_overlap](../../configs/forge/tasks/vector_overlap.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [vector_spiral](../../configs/forge/tasks/vector_spiral.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [img_stripes2](../../configs/forge/tasks/img_stripes2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_bars4](../../configs/forge/tasks/img_bars4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_blobs4](../../configs/forge/tasks/img_blobs4.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [img_intensity2](../../configs/forge/tasks/img_intensity2.json) | diagnostic | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [grid100](../../configs/forge/tasks/grid100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [rotated100](../../configs/forge/tasks/rotated100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [staggered100](../../configs/forge/tasks/staggered100.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## selected-family-coverage-diagnostic-v1

Declaration: [selected-family-coverage-diagnostic-v1](../../configs/forge/views/selected-family-coverage-diagnostic-v1.json); revision 1; goal: `clockfree_continuous`.

Declared calibration status: **undeclared**.

Declared evidence scope: `research_diagnostic`.

No published solution leaderboard for this view yet; task registration and related API media confer no candidate qualification.

### Tier 1: smoke

1 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## tier1_policy_coverage

Declaration: [tier1_policy_coverage](../../configs/forge/views/tier1_policy_coverage.json); revision 1; goal: `discriminator_stability`.

Declared calibration status: **undeclared**.

Candidate outcomes, metrics and measured costs: [leaderboard](technique-inventory.md).

### Tier 1: smoke

7 required.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / transfer_sustained | 1000 | 120 | — |
| [two_pole_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [unused_token_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [ae_gan_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [ring16_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 400 | 300 | — |
| [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / transfer_sustained | 20001 | 900 | — |
| [clockfree_audit_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json) | required | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-clockfree-audit-tier1-policy-selected-cloud-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## vaegan_coverage

Declaration: [vaegan_coverage](../../configs/forge/views/vaegan_coverage.json); revision 1; goal: `vaegan_coverage`.

Declared calibration status: **provisional**.

Two toy gates do not establish screen calibration or whole-suite adoption.

Declared evidence scope: `research_diagnostic`.

Candidate outcomes, metrics and measured costs: [leaderboard](leaderboards/vaegan_coverage.md).

### Tier 1: smoke

2 diagnostic.

| Task | Importance | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [model_vae_hard](../../configs/forge/tasks/model_vae_hard.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-api-model-vae-hard) | public_api_model_toy / transfer_sustained | 1200 | 300 | — |
| [model_vae_categorical](../../configs/forge/tasks/model_vae_categorical.json) | diagnostic | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-api-model-vae-categorical) | public_api_model_toy / transfer_sustained | 1200 | 300 | — |

### Tier 2: quality

0 tasks.

No tasks assigned.

### Tier 3: endurance

0 tasks.

No tasks assigned.

### Tier 4: stress_and_formulations

0 tasks.

No tasks assigned.

## Tasks unassigned to any view

These catalog tasks have no tier placement. Add an assignment to a view to include them in its policy.

| Task | Prior code path | Initialization / protocol | Experiment guide | Adapter / gate | Declared steps | Timeout (s) | Dependencies / shared execution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [ddgan_repair_bimodal_chain_v1](../../configs/forge/tasks/ddgan_repair_bimodal_chain_v1.json) | MoGParticlePrior (sigma=0.25) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-api-model-ddgan-bimodal-chain) | public_api_model_toy / transfer_sustained | 5000 | 600 | — |
| [five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / transfer_sustained | 20001 | 900 | — |
| [gaussian1d_acquisition](../../configs/forge/tasks/gaussian1d_acquisition.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / transfer_sustained | 1000 | 120 | — |
| [gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [gaussian1d_shallow_stability](../../configs/forge/tasks/gaussian1d_shallow_stability.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json) (checkpoint) |
| [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-grid100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [recipe-prior-refactor-legacy-ae_gan_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-ae_gan_hold.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ae-gan-hold) | transfer_behavior / transfer_sustained | 250 | 300 | — |
| [recipe-prior-refactor-legacy-clockfree_audit_measurement_v1](../../configs/forge/tasks/recipe-prior-refactor-legacy-clockfree_audit_measurement_v1.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-recipe-prior-refactor-legacy-clockfree-audit-measurement-v1) | clockfree_audit / clockfree_parity | 24 | 300 | — |
| [recipe-prior-refactor-legacy-cover_leftover](../../configs/forge/tasks/recipe-prior-refactor-legacy-cover_leftover.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-cover-leftover) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [recipe-prior-refactor-legacy-five_word_joint_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_hold | 4000 | 300 | [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json) (checkpoint) |
| [recipe-prior-refactor-legacy-five_word_joint_smoke](../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_smoke.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-five-word-joint) | word_joint / word_smoke | 20001 | 900 | — |
| [recipe-prior-refactor-legacy-gaussian1d_smoke](../../configs/forge/tasks/recipe-prior-refactor-legacy-gaussian1d_smoke.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_smoke | 1000 | 120 | — |
| [recipe-prior-refactor-legacy-gaussian1d_stability](../../configs/forge/tasks/recipe-prior-refactor-legacy-gaussian1d_stability.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-gaussian1d-acquisition) | transfer_vector / gaussian_stability | 6000 | 600 | [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json) (checkpoint) |
| [recipe-prior-refactor-legacy-grid100](../../configs/forge/tasks/recipe-prior-refactor-legacy-grid100.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-grid100) | native100 / native_accuracy | 7000 | 3600 | — |
| [recipe-prior-refactor-legacy-img_bars4](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_bars4.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-bars4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [recipe-prior-refactor-legacy-img_blobs4](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_blobs4.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-blobs4) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [recipe-prior-refactor-legacy-img_intensity2](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_intensity2.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-intensity2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [recipe-prior-refactor-legacy-img_stripes2](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_stripes2.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-img-stripes2) | transfer_image / transfer_sustained | 600 | 1800 | — |
| [recipe-prior-refactor-legacy-mid_scale_identity](../../configs/forge/tasks/recipe-prior-refactor-legacy-mid_scale_identity.json) | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mid-scale-identity) | transfer_behavior / transfer_sustained | 800 | 1800 | — |
| [recipe-prior-refactor-legacy-mode_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-mode_hold.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | transfer_behavior / transfer_sustained | 1200 | 1800 | — |
| [recipe-prior-refactor-legacy-residual_student](../../configs/forge/tasks/recipe-prior-refactor-legacy-residual_student.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-residual-student) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [recipe-prior-refactor-legacy-ring16_acquisition](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json) | MoGParticlePrior (sigma=0.1) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-ring16-acquisition) | transfer_vector / transfer_sustained | 1600 | 300 | — |
| [recipe-prior-refactor-legacy-ring_extension](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_extension.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_extension | up to 7500 total; 300 extension | 3600 | [ring_hold](../../configs/forge/tasks/ring_hold.json) (checkpoint); group: ring_endurance (uninterrupted) |
| [recipe-prior-refactor-legacy-ring_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-mode-hold) | ring_endurance / ring_hold | up to 7500 total | 3600 | [mode_hold](../../configs/forge/tasks/mode_hold.json) (gate); group: ring_endurance (uninterrupted) |
| [recipe-prior-refactor-legacy-rotated100](../../configs/forge/tasks/recipe-prior-refactor-legacy-rotated100.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-rotated100) | native100 / native_accuracy | 7000 | 3600 | — |
| [recipe-prior-refactor-legacy-staggered100](../../configs/forge/tasks/recipe-prior-refactor-legacy-staggered100.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-staggered100) | native100 / native_accuracy | 7000 | 3600 | — |
| [recipe-prior-refactor-legacy-trajectory](../../configs/forge/tasks/recipe-prior-refactor-legacy-trajectory.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-trajectory) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [recipe-prior-refactor-legacy-two_pole](../../configs/forge/tasks/recipe-prior-refactor-legacy-two_pole.json) | ParticlePrior (sigma=0) | deterministic_orthogonal; fixed: {"critic": "stored_host_weights", "particles": "zeros"}; screening | [Question, results, GIFs](#experiment-two-pole) | transfer_behavior / transfer_sustained | 80 | 300 | — |
| [recipe-prior-refactor-legacy-unipolar](../../configs/forge/tasks/recipe-prior-refactor-legacy-unipolar.json) | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unipolar) | transfer_behavior / transfer_sustained | 400 | 1800 | — |
| [recipe-prior-refactor-legacy-unused_token_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-unused_token_hold.json) | ParticlePrior (sigma=0; not sampled) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-unused-token-hold) | transfer_behavior / transfer_sustained | 200 | 300 | — |
| [recipe-prior-refactor-legacy-vector_anisotropic](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_anisotropic.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-anisotropic) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [recipe-prior-refactor-legacy-vector_overlap](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_overlap.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-overlap) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [recipe-prior-refactor-legacy-vector_spiral](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_spiral.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-spiral) | transfer_vector / transfer_sustained | 1600 | 1800 | — |
| [recipe-prior-refactor-legacy-vector_two_broad](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_two_broad.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-two-broad) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [recipe-prior-refactor-legacy-vector_unequal_mass](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_unequal_mass.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-mass) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [recipe-prior-refactor-legacy-vector_unequal_width](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_unequal_width.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; screening | [Question, results, GIFs](#experiment-vector-unequal-width) | transfer_vector / transfer_sustained | 1200 | 1800 | — |
| [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-rotated100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |
| [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json) | MoGParticlePrior (sigma=0.025) | deterministic_orthogonal; component policy; screening | [Question, results, GIFs](#experiment-staggered100) | native100_continuation / native_accuracy | 14000 total; 7000 additional | 7200 | [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json) (checkpoint); [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json) (gate) |

## Experiment guides

Task variants share a guide when their declarations name the same host or problem. This grouping is for navigation; it does not assert matching scientific contracts. Public-API variants join only through their explicit retained question IDs.

### Experiment: ae-gan-hold

Checks reconstruction/identity and an acquired adversarial edit during the declared hold.

Forge declarations: [ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json), [ae_gan_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ae_gan_hold_policy_cloud_coverage_v1.json), [ae_gan_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json), [recipe-prior-refactor-legacy-ae_gan_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-ae_gan_hold.json).

Declared Forge numerical gates and sampling:

[ae_gan_hold](../../configs/forge/tasks/ae_gan_hold.json)

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = encoder, generator, prior, discriminator; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | generated_and_reconstructed_prior_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

[ae_gan_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ae_gan_hold_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_generated_and_reconstructed_prior_with_scheduled_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | public_recipe_schedule |

[ae_gan_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ae_gan_hold_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_generated_and_reconstructed_prior_with_scheduled_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | public_recipe_schedule |

[recipe-prior-refactor-legacy-ae_gan_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-ae_gan_hold.json)

| Metric | Required bound |
| --- | --- |
| recon_mse | <= 0.05 |
| hold | <= 0.35 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = encoder, generator, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | generated_and_reconstructed_prior_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ae_gan_hold | [Atlas](../../configs/forge/ideas/atlas-particle-normal-v4.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / 6261a5458c1f / 767710fe7bd0 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [K3P](../../configs/forge/ideas/recipe-prior-refactor-k3p-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 3c8a0c05c790 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [KA2](../../configs/forge/ideas/recipe-prior-refactor-ka2-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / fb4738652dc8 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [R1/R2](../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / a266b096fb60 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [GAN v3 release 0.7](../../configs/forge/ideas/recipe-prior-refactor-release07-gan-v3-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 1accaa098064 | [source-bound receipt index](technique-inventory.json) |
| ae_gan_hold | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ae-anchor-hold](../toy_audit/api_contract/media/api-ae-anchor-hold.gif) | Reconstruct the noisy two-anchor inputs while the independently sampled prior covers both anchors Scope: New public Recipe.encode particle AE with reconstruction weight1 and applied GAN/KA2; independently sampled prior quality is separately scored. | MoGParticlePrior (sigma_rel=0.025) | COMPLETE / FAIL; 250/250 updates; quality_fraction, quality_mass_tv, last 5 post-update metric observations do not all pass | ae_gan / cpu / 3efb003d4c86 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [api-model-ae-gan-multimodal](../toy_audit/api_contract/model_modes/media/api-model-ae-gan-multimodal.gif) | Reconstruct four equally weighted noisy corner Gaussians and independently generate their modes, masses and within-mode shapes. Scope: Four equally weighted Gaussians at (+/-1,+/-1), sigma=.08; independently drawn target and prior samples. Reconstruction and all prior gates must pass together. | MoGParticlePrior (sigma_rel=0) | COMPLETE / PASS; 1200/1200 updates; recomputed complete live curve and terminal suffix | bispf / cpu / 1e57b5216ea3 | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |
| [api-model-ae-reconstruction](../toy_audit/api_contract/model_modes/media/api-model-ae-reconstruction.gif) | Reconstruct four equally weighted noisy corner Gaussians with public deterministic particle-AE routing and bounded offsets; explicit reconstruction-only objective control. Scope: Four equally weighted Gaussians at (+/-1,+/-1), sigma=.08; independently drawn target and prior samples. Explicit reconstruction-only objective control; independent generation is measured but not required. | MoGParticlePrior (sigma_rel=0) | COMPLETE / PASS; 1200/1200 updates; recomputed complete live curve and terminal suffix | bispf / cpu / 1e57b5216ea3 | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: alice-spiral-bigan

Continuous 2D Archimedes spiral joint (G(z), E(x), D(x, z)) with ALICE cycle-consistency regularization and MoGParticlePrior(sigma=0.06).

Forge declarations: [alice_spiral_bigan](../../configs/forge/tasks/alice_spiral_bigan.json).

Declared Forge numerical gates and sampling:

[alice_spiral_bigan](../../configs/forge/tasks/alice_spiral_bigan.json)

| Metric | Required bound |
| --- | --- |
| arc_bins | >= 15 |
| hq | >= 0.85 |
| mass_tv | <= 0.16 |
| recon_rmse | <= 0.14 |
| latent_recon_rmse | <= 0.25 |
| arc_consistency | >= 0.9 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.06) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: api-model-ae-ddgan-bimodal-chain

Reconstruct held-out inputs through a learned particle AE and four public DDGAN reverse steps while preserving independent generation and conditional-transition quality

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/ae-ddgan.md).

Forge declarations: [model_ae_ddgan_bimodal_chain](../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json).

Declared Forge numerical gates and sampling:

[model_ae_ddgan_bimodal_chain](../../configs/forge/tasks/model_ae_ddgan_bimodal_chain.json)

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.32 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.3 |
| final_covariance_eigen_max | <= 2.6 |
| t1_q0_projection_cdf | <= 0.75 |
| t1_q0_normalized_mean | <= 1.5 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 4.5 |
| t1_q1_projection_cdf | <= 0.75 |
| t1_q1_normalized_mean | <= 1.5 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 4.5 |
| t2_q0_projection_cdf | <= 0.28 |
| t2_q0_normalized_mean | <= 0.7 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.28 |
| t2_q1_normalized_mean | <= 0.7 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.22 |
| t3_q0_normalized_mean | <= 0.5 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.22 |
| t3_q1_normalized_mean | <= 0.5 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.9 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 1.5 |
| final_local_covariance_min | >= 0.3 |
| final_local_covariance_max | <= 2.05 |
| paired_nmse | <= 0.1 |
| paired_worst_axis_nrmse | <= 0.25 |
| paired_shuffled_gain | >= 3 |
| paired_reverse_steps | == 4 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.25) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-ae-ddgan-bimodal-chain](../toy_audit/api_contract/model_modes/media/api-model-ae-ddgan-bimodal-chain.gif) | Reconstruct held-out inputs through a learned particle AE and four public DDGAN reverse steps while preserving independent generation and conditional-transition quality Scope: New scoped public ae_ddgan composition: same analytic target, schedule, G/D architecture and prior as the DDGAN task, plus a learned 2x32 encoder. Clean paired reconstruction and independent-prior generation are separate cohorts. Historical denoising results retain their identities. | MoGParticlePrior (sigma_rel=0) | SOFTWARE_PREFIX_ONLY / BLOCKED; 3/5000 updates; model_ae_ddgan_gaussian_chain, model_ddgan_gaussian_chain, model_vae_hard | bispf / cpu / ba35fb2b55bb | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: api-model-ae-ddgan-gaussian-chain

Reconstruct held-out inputs through a learned particle AE and four public DDGAN reverse steps while preserving independent generation and conditional-transition quality

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/ae-ddgan.md).

Forge declarations: [model_ae_ddgan_gaussian_chain](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).

Declared Forge numerical gates and sampling:

[model_ae_ddgan_gaussian_chain](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json)

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.2 |
| final_normalized_mean | <= 0.45 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.26 |
| t1_q0_normalized_mean | <= 0.65 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 1.8 |
| t1_q1_projection_cdf | <= 0.26 |
| t1_q1_normalized_mean | <= 0.65 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 1.8 |
| t2_q0_projection_cdf | <= 0.2 |
| t2_q0_normalized_mean | <= 0.35 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.2 |
| t2_q1_normalized_mean | <= 0.35 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| paired_nmse | <= 0.1 |
| paired_worst_axis_nrmse | <= 0.25 |
| paired_shuffled_gain | >= 3 |
| paired_reverse_steps | == 4 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.25) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-ae-ddgan-gaussian-chain](../toy_audit/api_contract/model_modes/media/api-model-ae-ddgan-gaussian-chain.gif) | Reconstruct held-out inputs through a learned particle AE and four public DDGAN reverse steps while preserving independent generation and conditional-transition quality Scope: New scoped public ae_ddgan composition: same analytic target, schedule, G/D architecture and prior as the DDGAN task, plus a learned 2x32 encoder. Clean paired reconstruction and independent-prior generation are separate cohorts. Historical denoising results retain their identities. | MoGParticlePrior (sigma_rel=0) | COMPLETE / FAIL; 2500/2500 updates; final_covariance_eigen_max, final_normalized_mean, final_projection_cdf, t1_q0_covariance_eigen_min, t1_q0_normalized_mean, t1_q0_projection_cdf, t1_q1_covariance_eigen_min, t1_q1_normalized_mean, t1_q1_projection_cdf, t2_q0_normalized_mean, t2_q0_projection_cdf, last 5 post-update metric observations do not all pass | bispf / cpu / ba35fb2b55bb | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: api-model-ddgan-bimodal-chain

Learn conditional reverse transitions and the final ancestral distribution through the public DDGAN API; explicitly scoped GPU0 research diagnostic of original gates, no ordinary qualification

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/ddgan.md), [experiment readout](ddgan-conditional-repair/README.md).

Forge declarations: [ddgan_repair_bimodal_chain_v1](../../configs/forge/tasks/ddgan_repair_bimodal_chain_v1.json), [ddgan_repair_bimodal_chain_wall_v2](../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json), [model_ddgan_bimodal_chain](../../configs/forge/tasks/model_ddgan_bimodal_chain.json).

Declared Forge numerical gates and sampling:

[ddgan_repair_bimodal_chain_v1](../../configs/forge/tasks/ddgan_repair_bimodal_chain_v1.json), [ddgan_repair_bimodal_chain_wall_v2](../../configs/forge/tasks/ddgan_repair_bimodal_chain_wall_v2.json)

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.12 |
| t1_q0_normalized_mean | <= 0.25 |
| t1_q0_covariance_eigen_min | >= 0.65 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.12 |
| t1_q1_normalized_mean | <= 0.25 |
| t1_q1_covariance_eigen_min | >= 0.65 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.12 |
| t2_q0_normalized_mean | <= 0.25 |
| t2_q0_covariance_eigen_min | >= 0.65 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.12 |
| t2_q1_normalized_mean | <= 0.25 |
| t2_q1_covariance_eigen_min | >= 0.65 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.95 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 0.5 |
| final_local_covariance_min | >= 0.5 |
| final_local_covariance_max | <= 1.5 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.25) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[model_ddgan_bimodal_chain](../../configs/forge/tasks/model_ddgan_bimodal_chain.json)

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.32 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.3 |
| final_covariance_eigen_max | <= 2.6 |
| t1_q0_projection_cdf | <= 0.75 |
| t1_q0_normalized_mean | <= 1.5 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 4.5 |
| t1_q1_projection_cdf | <= 0.75 |
| t1_q1_normalized_mean | <= 1.5 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 4.5 |
| t2_q0_projection_cdf | <= 0.28 |
| t2_q0_normalized_mean | <= 0.7 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.28 |
| t2_q1_normalized_mean | <= 0.7 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.22 |
| t3_q0_normalized_mean | <= 0.5 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.22 |
| t3_q1_normalized_mean | <= 0.5 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |
| final_hq | >= 0.9 |
| final_mode_mass_tv | <= 0.1 |
| final_local_mean_sigma | <= 1.5 |
| final_local_covariance_min | >= 0.3 |
| final_local_covariance_max | <= 2.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.25) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-ddgan-bimodal-chain](../toy_audit/api_contract/model_modes/media/api-model-ddgan-bimodal-chain.gif) | Learn conditional reverse transitions and the final ancestral distribution through the public DDGAN API Scope: New small analytic bimodal target, 2x32 MLPs and clean live serving. Actual forward-pair transition training and four-step reverse sampling; no historical 100-Gaussian or direct-posterior qualification. | MoGParticlePrior (sigma_rel=0) | SOFTWARE_PREFIX_ONLY / BLOCKED; 3/5000 updates; model_ae_ddgan_gaussian_chain, model_ddgan_gaussian_chain, model_vae_hard | bispf / cpu / ba35fb2b55bb | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: api-model-ddgan-gaussian-chain

Learn conditional reverse transitions and the final ancestral distribution through the public DDGAN API; explicitly scoped GPU0 research diagnostic of original gates, no ordinary qualification

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/ddgan.md), [experiment readout](ddgan-conditional-repair/README.md).

Forge declarations: [ddgan_repair_gaussian_chain_v1](../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json), [model_ddgan_gaussian_chain](../../configs/forge/tasks/model_ddgan_gaussian_chain.json).

Declared Forge numerical gates and sampling:

[ddgan_repair_gaussian_chain_v1](../../configs/forge/tasks/ddgan_repair_gaussian_chain_v1.json)

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.12 |
| t1_q0_normalized_mean | <= 0.25 |
| t1_q0_covariance_eigen_min | >= 0.65 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.12 |
| t1_q1_normalized_mean | <= 0.25 |
| t1_q1_covariance_eigen_min | >= 0.65 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.12 |
| t2_q0_normalized_mean | <= 0.25 |
| t2_q0_covariance_eigen_min | >= 0.65 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.12 |
| t2_q1_normalized_mean | <= 0.25 |
| t2_q1_covariance_eigen_min | >= 0.65 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.25) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[model_ddgan_gaussian_chain](../../configs/forge/tasks/model_ddgan_gaussian_chain.json)

| Metric | Required bound |
| --- | --- |
| final_projection_cdf | <= 0.12 |
| final_normalized_mean | <= 0.25 |
| final_covariance_eigen_min | >= 0.65 |
| final_covariance_eigen_max | <= 1.4 |
| t1_q0_projection_cdf | <= 0.26 |
| t1_q0_normalized_mean | <= 0.45 |
| t1_q0_covariance_eigen_min | >= 0.4 |
| t1_q0_covariance_eigen_max | <= 1.4 |
| t1_q1_projection_cdf | <= 0.26 |
| t1_q1_normalized_mean | <= 0.45 |
| t1_q1_covariance_eigen_min | >= 0.4 |
| t1_q1_covariance_eigen_max | <= 1.4 |
| t2_q0_projection_cdf | <= 0.2 |
| t2_q0_normalized_mean | <= 0.35 |
| t2_q0_covariance_eigen_min | >= 0.6 |
| t2_q0_covariance_eigen_max | <= 1.4 |
| t2_q1_projection_cdf | <= 0.2 |
| t2_q1_normalized_mean | <= 0.35 |
| t2_q1_covariance_eigen_min | >= 0.6 |
| t2_q1_covariance_eigen_max | <= 1.4 |
| t3_q0_projection_cdf | <= 0.12 |
| t3_q0_normalized_mean | <= 0.25 |
| t3_q0_covariance_eigen_min | >= 0.65 |
| t3_q0_covariance_eigen_max | <= 1.4 |
| t3_q1_projection_cdf | <= 0.12 |
| t3_q1_normalized_mean | <= 0.25 |
| t3_q1_covariance_eigen_min | >= 0.65 |
| t3_q1_covariance_eigen_max | <= 1.4 |
| t4_q0_projection_cdf | <= 0.12 |
| t4_q0_normalized_mean | <= 0.25 |
| t4_q0_covariance_eigen_min | >= 0.65 |
| t4_q0_covariance_eigen_max | <= 1.4 |
| t4_q1_projection_cdf | <= 0.12 |
| t4_q1_normalized_mean | <= 0.25 |
| t4_q1_covariance_eigen_min | >= 0.65 |
| t4_q1_covariance_eigen_max | <= 1.4 |
| panel_count | == 8 |
| samples_per_panel | >= 2048 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.25) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-ddgan-gaussian-chain](../toy_audit/api_contract/model_modes/media/api-model-ddgan-gaussian-chain.gif) | Learn conditional reverse transitions and the final ancestral distribution through the public DDGAN API Scope: New small analytic gaussian target, 2x32 MLPs and clean live serving. Actual forward-pair transition training and four-step reverse sampling; no historical 100-Gaussian or direct-posterior qualification. | MoGParticlePrior (sigma_rel=0) | COMPLETE / FAIL; 2500/2500 updates; final_covariance_eigen_min, final_normalized_mean, final_projection_cdf, t1_q0_covariance_eigen_max, t1_q0_normalized_mean, t1_q0_projection_cdf, t1_q1_covariance_eigen_max, t1_q1_normalized_mean, t1_q1_projection_cdf, t2_q0_normalized_mean, t2_q0_projection_cdf, t2_q1_covariance_eigen_min, t2_q1_normalized_mean, t2_q1_projection_cdf, last 5 post-update metric observations do not all pass | bispf / cpu / 48177d41bb0b | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: api-model-vae-categorical

Can a genuine particle VAE retain the input and posterior noise while independently generating both target modes? New constrained two-mode seed-0 public-component cohort; does not reproduce or qualify historical 100-mode results. Hard routing has constant log(K) KL and biased straight-through gradients; categorical routing uses the public leave-one-out score estimator and a variable KL. A stochastic no-KL AE is not a VAE.

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/vae.md).

Forge declarations: [model_vae_categorical](../../configs/forge/tasks/model_vae_categorical.json).

Declared Forge numerical gates and sampling:

[model_vae_categorical](../../configs/forge/tasks/model_vae_categorical.json)

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.05 |
| shuffle_reconstruction_ratio | >= 5 |
| posterior_mutual_information | >= 0.3 |
| posterior_noise_variance_ratio | >= 0.8 |
| posterior_noise_variance_ratio | <= 1.2 |
| kl_identity_error | <= 1e-05 |
| elbo_identity_error | <= 0.0001 |
| decoder_mean_hq | >= 0.85 |
| decoder_mean_mass_tv | <= 0.15 |
| decoder_mean_modes | >= 2 |
| likelihood_hq | >= 0.85 |
| likelihood_mass_tv | <= 0.15 |
| likelihood_modes | >= 2 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = encoder, generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-vae-categorical](../toy_audit/api_contract/model_modes/media/api-model-vae-categorical.gif) | Can a genuine particle VAE retain the input and posterior noise while independently generating both target modes? Scope: New constrained two-mode seed-0 public-component cohort; does not reproduce or qualify historical 100-mode results. Hard routing has constant log(K) KL and biased straight-through gradients; categorical routing uses the public leave-one-out score estimator and a variable KL. A stochastic no-KL AE is not a VAE. | MoGParticlePrior (sigma_rel=0) | COMPLETE / PASS; 1200/1200 updates; recomputed complete live curve and terminal suffix | bispf / cpu / 48177d41bb0b | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: api-model-vae-hard

Can a genuine particle VAE retain the input and posterior noise while independently generating both target modes? New constrained two-mode seed-0 public-component cohort; does not reproduce or qualify historical 100-mode results. Hard routing has constant log(K) KL and biased straight-through gradients; categorical routing uses the public leave-one-out score estimator and a variable KL. A stochastic no-KL AE is not a VAE.

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/vae.md).

Forge declarations: [model_vae_hard](../../configs/forge/tasks/model_vae_hard.json).

Declared Forge numerical gates and sampling:

[model_vae_hard](../../configs/forge/tasks/model_vae_hard.json)

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.05 |
| shuffle_reconstruction_ratio | >= 5 |
| posterior_mutual_information | >= 0.3 |
| posterior_noise_variance_ratio | >= 0.8 |
| posterior_noise_variance_ratio | <= 1.2 |
| kl_identity_error | <= 1e-05 |
| elbo_identity_error | <= 0.0001 |
| decoder_mean_hq | >= 0.85 |
| decoder_mean_mass_tv | <= 0.15 |
| decoder_mean_modes | >= 2 |
| likelihood_hq | >= 0.85 |
| likelihood_mass_tv | <= 0.15 |
| likelihood_modes | >= 2 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = encoder, generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-vae-hard](../toy_audit/api_contract/model_modes/media/api-model-vae-hard.gif) | Can a genuine particle VAE retain the input and posterior noise while independently generating both target modes? Scope: New constrained two-mode seed-0 public-component cohort; does not reproduce or qualify historical 100-mode results. Hard routing has constant log(K) KL and biased straight-through gradients; categorical routing uses the public leave-one-out score estimator and a variable KL. A stochastic no-KL AE is not a VAE. | MoGParticlePrior (sigma_rel=0) | COMPLETE / FAIL; 1200/1200 updates; last 5 post-update metric observations do not all pass | bispf / cpu / 48177d41bb0b | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: cgan-overlapping-classes

Conditional GAN with Projection Discriminator / UCD on 4 overlapping 2D Gaussian classes (sigma=0.35), testing preservation of within-class variance and central overlap density against AC-GAN boundary repulsion.

Forge declarations: [cgan_overlapping_classes](../../configs/forge/tasks/cgan_overlapping_classes.json).

Declared Forge numerical gates and sampling:

[cgan_overlapping_classes](../../configs/forge/tasks/cgan_overlapping_classes.json)

| Metric | Required bound |
| --- | --- |
| classes_covered | == 4 |
| class_mass_tv | <= 0.08 |
| mean_center_err | <= 0.12 |
| min_cov_eig_ratio | >= 0.6 |
| max_cov_eig_ratio | <= 1.5 |
| overlap_density_ratio | >= 0.55 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit

Check saved public trainer state under step_label, horizon, evaluation_cadence, restart perturbations.

Forge declarations: [clockfree_audit](../../configs/forge/tasks/clockfree_audit.json).

Declared Forge numerical gates and sampling:

[clockfree_audit](../../configs/forge/tasks/clockfree_audit.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit-measurement-v1

Measure the original four clock/state parity conditions and zero-dependency condition for an explicitly scheduled recipe; diagnostic FAIL grants no clock-free claim or qualification.

Forge declarations: [clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json).

Declared Forge numerical gates and sampling:

[clockfree_audit_measurement_v1](../../configs/forge/tasks/clockfree_audit_measurement_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit-measurement-v1-policy-cloud-coverage-v1

Measure the original four clock/state parity conditions and zero-dependency condition for an explicitly scheduled recipe; diagnostic FAIL grants no clock-free claim or qualification.

Forge declarations: [clockfree_audit_measurement_v1_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_measurement_v1_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[clockfree_audit_measurement_v1_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_measurement_v1_policy_cloud_coverage_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit-policy-cloud-coverage-v1

Check saved public trainer state under step_label, horizon, evaluation_cadence, restart perturbations.

Forge declarations: [clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[clockfree_audit_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/clockfree_audit_policy_cloud_coverage_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: clockfree-audit-tier1-policy-selected-cloud-v1

Check saved public trainer state under step_label, horizon, evaluation_cadence, restart perturbations.

Forge declarations: [clockfree_audit_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[clockfree_audit_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/clockfree_audit_tier1_policy_selected_cloud_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: cover-leftover

Checks target coverage plus the separate unwanted-remainder/content constraints.

Forge declarations: [cover_leftover](../../configs/forge/tasks/cover_leftover.json), [cover_leftover_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/cover_leftover_policy_cloud_coverage_v1.json), [recipe-prior-refactor-legacy-cover_leftover](../../configs/forge/tasks/recipe-prior-refactor-legacy-cover_leftover.json).

Declared Forge numerical gates and sampling:

[cover_leftover](../../configs/forge/tasks/cover_leftover.json)

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[cover_leftover_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/cover_leftover_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_learned_parameter_measurement |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

[recipe-prior-refactor-legacy-cover_leftover](../../configs/forge/tasks/recipe-prior-refactor-legacy-cover_leftover.json)

| Metric | Required bound |
| --- | --- |
| u_kept | >= 0.85 |
| content_kept | >= 0.75 |
| leak_ratio | <= 0.2 |
| pole_rel_err_plus | <= 0.2 |
| pole_rel_err_minus | <= 0.2 |
| same_dir | <= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| cover_leftover | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| cover_leftover | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| cover_leftover | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| cover_leftover | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-guarded-leftover](../toy_audit/api_contract/media/api-guarded-leftover.gif) | Cover both signed poles while preserving content and removing the guarded leak Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 800/800 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: cramer-energy-overlap

CramerGAN two-sample unbiased energy distance (Bellemare et al., NeurIPS 2017) on 9-mode 3x3 overlapping heterogeneous-mass 2D Gaussian mixture.

Forge declarations: [cramer_energy_overlap](../../configs/forge/tasks/cramer_energy_overlap.json).

Declared Forge numerical gates and sampling:

[cramer_energy_overlap](../../configs/forge/tasks/cramer_energy_overlap.json)

| Metric | Required bound |
| --- | --- |
| modes | == 9 |
| hq | >= 0.88 |
| mass_tv | <= 0.1 |
| energy_distance | <= 0.06 |
| min_cov_eig_ratio | >= 0.35 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: cyclegan-unpaired-constellation

Unpaired 2D domain translation and mode-level bijection (G_xy, G_yx, D_y, D_x) between an 8-mode ring (R=2.5, sigma=0.08) and an 8-mode square-perimeter constellation ({-2, 0, +2}^2 \ {(0, 0)}, sigma=0.08) with cycle-consistency regularization (Zhu et al. ICCV 2017; Kim et al. ICML 2017).

Forge declarations: [cyclegan_unpaired_constellation](../../configs/forge/tasks/cyclegan_unpaired_constellation.json).

Declared Forge numerical gates and sampling:

[cyclegan_unpaired_constellation](../../configs/forge/tasks/cyclegan_unpaired_constellation.json)

| Metric | Required bound |
| --- | --- |
| modes_y | >= 7 |
| modes_x | >= 7 |
| hq_y | >= 0.85 |
| hq_x | >= 0.85 |
| mass_tv_y | <= 0.15 |
| cycle_rmse | <= 0.12 |
| bijection_ber | <= 0.15 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.08) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-b64

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_b64](../../configs/forge/tasks/dcgan_performance_b64.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_b64](../../configs/forge/tasks/dcgan_performance_b64.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-cpu-smoke

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_cpu_smoke](../../configs/forge/tasks/dcgan_performance_cpu_smoke.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_cpu_smoke](../../configs/forge/tasks/dcgan_performance_cpu_smoke.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-r128

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_r128](../../configs/forge/tasks/dcgan_performance_r128.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_r128](../../configs/forge/tasks/dcgan_performance_r128.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-r256

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_r256](../../configs/forge/tasks/dcgan_performance_r256.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_r256](../../configs/forge/tasks/dcgan_performance_r256.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-r32

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_r32](../../configs/forge/tasks/dcgan_performance_r32.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_r32](../../configs/forge/tasks/dcgan_performance_r32.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-r64

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_r64](../../configs/forge/tasks/dcgan_performance_r64.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_r64](../../configs/forge/tasks/dcgan_performance_r64.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: dcgan-performance-w128

Execute the declared image_training_performance gate.

Forge declarations: [dcgan_performance_w128](../../configs/forge/tasks/dcgan_performance_w128.json).

Declared Forge numerical gates and sampling:

[dcgan_performance_w128](../../configs/forge/tasks/dcgan_performance_w128.json)


| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | training_update_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: five-word-joint

Can a joint BiGAN generator, encoder and critic acquire five equally likely canonical words and reconstruct every correctly paired input with confident token probabilities, including padding?

Explanation, interpretation and reproduction: [experiment readout](five-word-joint/README.md), [experiment readout](five-word-tier-split/README.md).

Forge declarations: [five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json), [five_word_joint_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json), [five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json), [five_word_joint_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_hold_policy_cloud_coverage_v1.json), [five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json), [five_word_joint_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_smoke_policy_cloud_coverage_v1.json), [recipe-prior-refactor-legacy-five_word_joint_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json), [recipe-prior-refactor-legacy-five_word_joint_smoke](../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_smoke.json).

Declared Forge numerical gates and sampling:

[five_word_joint_acquisition](../../configs/forge/tasks/five_word_joint_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[five_word_joint_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/five_word_joint_acquisition_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[five_word_joint_hold](../../configs/forge/tasks/five_word_joint_hold.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = False.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[five_word_joint_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_hold_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: exact optimizer updates = False; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[five_word_joint_smoke](../../configs/forge/tasks/five_word_joint_smoke.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: finite state = True; optimizer roles = generator, encoder, prior, discriminator; mechanism exercised = True; rng isolation = True; exact optimizer updates = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[five_word_joint_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/five_word_joint_smoke_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[recipe-prior-refactor-legacy-five_word_joint_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_hold.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: exact optimizer updates = False; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[recipe-prior-refactor-legacy-five_word_joint_smoke](../../configs/forge/tasks/recipe-prior-refactor-legacy-five_word_joint_smoke.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 1024 |
| quality_fraction | >= 0.95 |
| modes | == 5 |
| mass_tv | <= 0.1 |
| reconstruction_exact | == 1 |
| minimum_reconstruction_token_probability | >= 0.9 |

Execution guards: exact optimizer updates = True; finite state = True; mechanism exercised = True; optimizer roles = generator, encoder, prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | generated_and_paired_reconstructed_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| five_word_joint_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_hold | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_hold | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_hold | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [Atlas](../../configs/forge/ideas/atlas-particle-normal-v4.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 6261a5458c1f / 767710fe7bd0 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [K3P](../../configs/forge/ideas/recipe-prior-refactor-k3p-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 3c8a0c05c790 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [KA2](../../configs/forge/ideas/recipe-prior-refactor-ka2-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / fb4738652dc8 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [R1/R2](../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / a266b096fb60 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [GAN v3 release 0.7](../../configs/forge/ideas/recipe-prior-refactor-release07-gan-v3-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 1accaa098064 | [source-bound receipt index](technique-inventory.json) |
| five_word_joint_smoke | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [forge-five-word-joint-api-demo-v1](../toy_audit/api_contract/five_word_joint/goal.gif) | Can a joint BiGAN generator, encoder and critic acquire five equally likely canonical words and reconstruct every correctly paired input with confident token probabilities, including padding? Scope: One bounded shared-API integration demonstration, not an ordinary Forge run or release qualification. Score the declared bounds honestly at 32 updates and grade the evidence INCOMPLETE against the 20,001-update task. | ParticlePrior (sigma=0) | COMPLETE / INCOMPLETE; 32/20001 updates; quality_fraction, modes, mass_tv, reconstruction_exact, minimum_reconstruction_token_probability | ka2 / cpu / 997c7f01b99a | [definition](../toy_audit/api_contract/five_word_joint/publication.json); [readout](../toy_audit/api_contract/five_word_joint/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_joint/publication.json) |
| [image-five-word-joint-hold-confirmed-v1](../toy_audit/api_contract/five_word_smoke_hold/media/five_word_joint_hold.gif) | Continue this candidate's earliest confirmed five-word acquisition state for 4,000 updates. Every scheduled generation and inverse check, including the exact restored state, must pass. Scope: task-only selected BCAP DualNorm verification; no ordinary family qualification | ParticlePrior (sigma=0) | COMPLETE / FAIL; 4000/4000 updates | bcap / unrecorded / 3c82db6c7b24 | [definition](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [readout](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_smoke_hold/publication.json) |
| [image-five-word-joint-smoke-confirmed-v1](../toy_audit/api_contract/five_word_smoke_hold/media/five_word_joint_smoke.gif) | Can public joint BiGAN training acquire all five words and confidently reconstruct every paired input at one independently confirmed scheduled state? Complete all 20,001 updates. Scope: task-only selected BCAP DualNorm verification; no ordinary family qualification | ParticlePrior (sigma=0) | COMPLETE / PASS; 20001/20001 updates | bcap / unrecorded / 3c82db6c7b24 | [definition](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [readout](../toy_audit/api_contract/five_word_smoke_hold/publication.json); [recipe and provenance](../toy_audit/api_contract/five_word_smoke_hold/publication.json) |
| [image-five-words-joint-ae](../toy_audit/api_contract/media/image-five-words-joint-ae.gif) | Generate the five equally likely canonical words with confident normalized token probabilities, and reconstruct each of the five matched inputs including underscore padding. Scope: Finite vocabulary apple/grape/lemon/melon/berry only. Joint BiGAN inverse reconstruction; no unseen words or natural-language generation. New API-policy variant, not reuse of historical EMA PASS. | ParticlePrior (sigma=0) | COMPLETE / PASS; 20001/20001 updates | ka2 / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: gaussian1d-acquisition

Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks?

Explanation, interpretation and reproduction: [experiment readout](gaussian-shallow/README.md), [experiment readout](gaussian-smoke-tier-split/README.md), [experiment readout](../toy_audit/api_contract/gaussian1d/README.md).

Forge declarations: [gaussian1d_acquisition](../../configs/forge/tasks/gaussian1d_acquisition.json), [gaussian1d_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json), [gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json), [gaussian1d_shallow_stability](../../configs/forge/tasks/gaussian1d_shallow_stability.json), [gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json), [gaussian1d_smoke_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_smoke_policy_cloud_coverage_as_served_v1.json), [gaussian1d_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_smoke_policy_cloud_coverage_v1.json), [gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json), [gaussian1d_stability_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_stability_policy_cloud_coverage_as_served_v1.json), [gaussian1d_stability_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_stability_policy_cloud_coverage_v1.json), [recipe-prior-refactor-legacy-gaussian1d_smoke](../../configs/forge/tasks/recipe-prior-refactor-legacy-gaussian1d_smoke.json), [recipe-prior-refactor-legacy-gaussian1d_stability](../../configs/forge/tasks/recipe-prior-refactor-legacy-gaussian1d_stability.json).

Declared Forge numerical gates and sampling:

[gaussian1d_acquisition](../../configs/forge/tasks/gaussian1d_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/gaussian1d_acquisition_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[gaussian1d_shallow_smoke](../../configs/forge/tasks/gaussian1d_shallow_smoke.json), [recipe-prior-refactor-legacy-gaussian1d_smoke](../../configs/forge/tasks/recipe-prior-refactor-legacy-gaussian1d_smoke.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_shallow_stability](../../configs/forge/tasks/gaussian1d_shallow_stability.json), [recipe-prior-refactor-legacy-gaussian1d_stability](../../configs/forge/tasks/recipe-prior-refactor-legacy-gaussian1d_stability.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_smoke](../../configs/forge/tasks/gaussian1d_smoke.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_smoke_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_smoke_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[gaussian1d_smoke_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_smoke_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[gaussian1d_stability](../../configs/forge/tasks/gaussian1d_stability.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; rng isolation = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[gaussian1d_stability_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/gaussian1d_stability_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[gaussian1d_stability_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/gaussian1d_stability_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator, prior; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| gaussian1d_smoke | [Atlas](../../configs/forge/ideas/atlas-particle-normal-v4.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 6261a5458c1f / 767710fe7bd0 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [K3P](../../configs/forge/ideas/recipe-prior-refactor-k3p-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 3c8a0c05c790 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [KA2](../../configs/forge/ideas/recipe-prior-refactor-ka2-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / fb4738652dc8 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [R1/R2](../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / a266b096fb60 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [GAN v3 release 0.7](../../configs/forge/ideas/recipe-prior-refactor-release07-gan-v3-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 1accaa098064 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_smoke | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_stability | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_stability | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_stability | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| gaussian1d_stability | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-gaussian1d-acquisition](../toy_audit/api_contract/gaussian1d/goal.gif) | Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks? Scope: Acquire N(2, .5^2) in 1,000 updates from random initialization; exact Gaussian CDF, location and width gates must pass at five terminal checks. Tier 1 remains provisional; this standalone API cohort grants no whole-view qualification. | MoGParticlePrior (sigma_rel=0) | COMPLETE / FAIL; 1000/1000 updates; cdf_ks <= 0.05, last 5 post-update metric observations do not all pass | k3p / cpu / 666642486d3c | [definition](../toy_audit/api_contract/gaussian1d/results.json); [readout](../toy_audit/api_contract/gaussian1d/results.json); [recipe and provenance](../toy_audit/api_contract/gaussian1d/results.json) |

### Experiment: grid100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [grid100](../../configs/forge/tasks/grid100.json), [grid100_14k](../../configs/forge/tasks/grid100_14k.json), [grid100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_14k_policy_cloud_coverage_as_served_v1.json), [grid100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_14k_policy_cloud_coverage_v1.json), [grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json), [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json), [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json), [grid100_mlp16](../../configs/forge/tasks/grid100_mlp16.json), [grid100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_policy_cloud_coverage_as_served_v1.json), [grid100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_policy_cloud_coverage_v1.json), [grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json), [grid100_res16_scaled_tanh](../../configs/forge/tasks/grid100_res16_scaled_tanh.json), [grid100_res16_tanh](../../configs/forge/tasks/grid100_res16_tanh.json), [recipe-prior-refactor-legacy-grid100](../../configs/forge/tasks/recipe-prior-refactor-legacy-grid100.json).

Declared Forge numerical gates and sampling:

[grid100](../../configs/forge/tasks/grid100.json), [grid100_14k](../../configs/forge/tasks/grid100_14k.json), [grid100_mlp16](../../configs/forge/tasks/grid100_mlp16.json), [grid100_res16_tanh](../../configs/forge/tasks/grid100_res16_tanh.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[grid100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_14k_policy_cloud_coverage_as_served_v1.json), [grid100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/grid100_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[grid100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_14k_policy_cloud_coverage_v1.json), [grid100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/grid100_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[grid100_affine_paired_laws_v1](../../configs/forge/tasks/grid100_affine_paired_laws_v1.json), [grid100_affine_square_named_v1](../../configs/forge/tasks/grid100_affine_square_named_v1.json), [grid100_affine_square_named_v1_14k](../../configs/forge/tasks/grid100_affine_square_named_v1_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[grid100_release07_cloud_named_v1](../../configs/forge/tasks/grid100_release07_cloud_named_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[grid100_res16_scaled_tanh](../../configs/forge/tasks/grid100_res16_scaled_tanh.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[recipe-prior-refactor-legacy-grid100](../../configs/forge/tasks/recipe-prior-refactor-legacy-grid100.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| grid100 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| grid100 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| grid100 | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| grid100 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-grid100](../toy_audit/api_contract/media/api-grid100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-bars4

Healthy location transfer: four horizontal/vertical bar positions test spatial coverage.

Forge declarations: [img_bars4](../../configs/forge/tasks/img_bars4.json), [img_bars4_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_bars4_policy_cloud_coverage_as_served_v1.json), [img_bars4_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_bars4_policy_cloud_coverage_v1.json), [img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json), [img_bars4_w64_optimizer_diagnostic](../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json), [recipe-prior-refactor-legacy-img_bars4](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_bars4.json).

Declared Forge numerical gates and sampling:

[img_bars4](../../configs/forge/tasks/img_bars4.json), [img_bars4_w64_optimizer_diagnostic](../../configs/forge/tasks/img_bars4_w64_optimizer_diagnostic.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[img_bars4_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_bars4_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_enumerated_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[img_bars4_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_bars4_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_enumerated_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[img_bars4_residual16](../../configs/forge/tasks/img_bars4_residual16.json), [recipe-prior-refactor-legacy-img_bars4](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_bars4.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_bars4 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| img_bars4 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| img_bars4 | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| img_bars4 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_bars4-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_bars4-residual_upsample16.gif) | Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. Scope: Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_bars4-source-transpose12](../toy_audit/api_contract/media/image-develop-img_bars4-source-transpose12.gif) | Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. Scope: Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, modes, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-blobs4

Healthy location transfer: four small corner patches test localized quality and coverage.

Forge declarations: [img_blobs4](../../configs/forge/tasks/img_blobs4.json), [img_blobs4_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_blobs4_policy_cloud_coverage_as_served_v1.json), [img_blobs4_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_blobs4_policy_cloud_coverage_v1.json), [img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json), [recipe-prior-refactor-legacy-img_blobs4](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_blobs4.json).

Declared Forge numerical gates and sampling:

[img_blobs4](../../configs/forge/tasks/img_blobs4.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[img_blobs4_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_blobs4_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_enumerated_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[img_blobs4_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_blobs4_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_enumerated_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[img_blobs4_residual16](../../configs/forge/tasks/img_blobs4_residual16.json), [recipe-prior-refactor-legacy-img_blobs4](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_blobs4.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_blobs4 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| img_blobs4 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| img_blobs4 | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| img_blobs4 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_blobs4-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_blobs4-residual_upsample16.gif) | Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. Scope: Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_blobs4-source-transpose12](../toy_audit/api_contract/media/image-develop-img_blobs4-source-transpose12.gif) | Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. Scope: Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates; distribution_tv, finite_template_tv, hq, modes, last 5 post-update metric observations do not all pass | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-intensity2

Healthy photometric transfer: two patch intensities require intensity fidelity as well as support coverage.

Forge declarations: [img_intensity2](../../configs/forge/tasks/img_intensity2.json), [img_intensity2_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_intensity2_policy_cloud_coverage_as_served_v1.json), [img_intensity2_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_intensity2_policy_cloud_coverage_v1.json), [img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json), [recipe-prior-refactor-legacy-img_intensity2](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_intensity2.json).

Declared Forge numerical gates and sampling:

[img_intensity2](../../configs/forge/tasks/img_intensity2.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[img_intensity2_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_intensity2_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_enumerated_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[img_intensity2_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_intensity2_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_enumerated_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[img_intensity2_residual16](../../configs/forge/tasks/img_intensity2_residual16.json), [recipe-prior-refactor-legacy-img_intensity2](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_intensity2.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_intensity2 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| img_intensity2 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| img_intensity2 | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| img_intensity2 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_intensity2-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_intensity2-residual_upsample16.gif) | Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. Scope: Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_intensity2-source-transpose12](../toy_audit/api_contract/media/image-develop-img_intensity2-source-transpose12.gif) | Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. Scope: Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: img-stripes2

Healthy orientation transfer: two distinct stripe orientations with an adequately sized convolutional GAN.

Forge declarations: [img_stripes2](../../configs/forge/tasks/img_stripes2.json), [img_stripes2_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_stripes2_policy_cloud_coverage_as_served_v1.json), [img_stripes2_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_stripes2_policy_cloud_coverage_v1.json), [img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json), [img_stripes2_w64_optimizer_diagnostic](../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json), [recipe-prior-refactor-legacy-img_stripes2](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_stripes2.json).

Declared Forge numerical gates and sampling:

[img_stripes2](../../configs/forge/tasks/img_stripes2.json), [img_stripes2_w64_optimizer_diagnostic](../../configs/forge/tasks/img_stripes2_w64_optimizer_diagnostic.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[img_stripes2_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/img_stripes2_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_enumerated_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[img_stripes2_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/img_stripes2_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_enumerated_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[img_stripes2_residual16](../../configs/forge/tasks/img_stripes2_residual16.json), [recipe-prior-refactor-legacy-img_stripes2](../../configs/forge/tasks/recipe-prior-refactor-legacy-img_stripes2.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 2 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| img_stripes2 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| img_stripes2 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| img_stripes2 | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| img_stripes2 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [image-develop-img_stripes2-residual_upsample16](../toy_audit/api_contract/media/image-develop-img_stripes2-residual_upsample16.gif) | Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. Scope: Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [image-develop-img_stripes2-source-transpose12](../toy_audit/api_contract/media/image-develop-img_stripes2-source-transpose12.gif) | Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. Scope: Recover both centered horizontal and vertical stripes with pixel contrast and balanced output mass. | ParticlePrior (sigma=0) | COMPLETE / PASS; 600/600 updates | atlas / cpu / 39eff89a9223 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: infogan-anisotropic-modes

Variational mutual information InfoGAN (G([z, c_disc, c_cont]), D(x), Q(c\|x)) on 8 rotated anisotropic 2D Gaussians arranged on a ring (R=3.0, sigma_parallel=0.25, sigma_perp=0.05).

Forge declarations: [infogan_anisotropic_modes](../../configs/forge/tasks/infogan_anisotropic_modes.json).

Declared Forge numerical gates and sampling:

[infogan_anisotropic_modes](../../configs/forge/tasks/infogan_anisotropic_modes.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.85 |
| mass_tv | <= 0.12 |
| discrete_nmi | >= 0.85 |
| q_acc | >= 0.9 |
| continuous_r2 | >= 0.65 |

At least 5 consecutive passing terminal observations.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.05) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: interp-img-patch-shift

Learn 16 zero-sigma ParticlePrior codes on continuous 1x8x8 single-patch images without MoG smoothing so unseen latent k-NN chords translate one sharp unit-peak patch (peak>=0.78, rmse<=0.10) instead of double-exposed 0.5 ghosts.

Explanation, interpretation and reproduction: [experiment readout](particle-interpolation-stretch/README.md).

Forge declarations: [interp_img_patch_shift](../../configs/forge/tasks/interp_img_patch_shift.json).

Declared Forge numerical gates and sampling:

[interp_img_patch_shift](../../configs/forge/tasks/interp_img_patch_shift.json)

| Metric | Required bound |
| --- | --- |
| finite_fraction | == 1 |
| endpoint_rmse | <= 0.09 |
| interp_rmse | <= 0.1 |
| interp_peak | >= 0.78 |
| quadrant_coverage | == 4 |
| midpoint_novelty | >= 0.18 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-interp-img-patch-shift](../toy_audit/api_contract/interpolation/media/api-interp-img-patch-shift.gif) | Learn 16 zero-sigma ParticlePrior codes on continuous 8x8 single-patch images so unseen latent k-NN chords translate one sharp bright patch (peak>=0.78, rmse<=0.10) instead of double-exposed 0.5 ghosts. Scope: Public API stretch variant: zero-sigma ParticlePrior image latent interpolation | ParticlePrior (sigma=0) | COMPLETE / PASS; 800/800 updates | atlas / unrecorded / 6e617abd2d29 | [definition](../toy_audit/api_contract/interpolation/publication.json); [readout](../toy_audit/api_contract/interpolation/publication.json); [recipe and provenance](../toy_audit/api_contract/interpolation/publication.json) |

### Experiment: interp-saddle-barycentric

Learn 24 zero-sigma ParticlePrior locations on a non-harmonic curved 3D saddle sheet without MoG smoothing so unseen 2D latent k-NN triangle barycenters remain on the curved surface (mae<=0.09, hq>=0.78) without Voronoi snapping.

Explanation, interpretation and reproduction: [experiment readout](particle-interpolation-stretch/README.md).

Forge declarations: [interp_saddle_barycentric](../../configs/forge/tasks/interp_saddle_barycentric.json).

Declared Forge numerical gates and sampling:

[interp_saddle_barycentric](../../configs/forge/tasks/interp_saddle_barycentric.json)

| Metric | Required bound |
| --- | --- |
| finite_fraction | == 1 |
| endpoint_surface_mae | <= 0.08 |
| barycenter_surface_mae | <= 0.09 |
| barycenter_hq | >= 0.78 |
| xy_bins_covered | >= 12 |
| barycenter_novelty | >= 0.18 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-interp-saddle-barycentric](../toy_audit/api_contract/interpolation/media/api-interp-saddle-barycentric.gif) | Learn 24 zero-sigma ParticlePrior locations on a non-harmonic 3D saddle surface so unseen 2D latent k-NN triangle barycenters stay on the curved surface (mae<=0.09, hq>=0.78) without snapping. Scope: Public API stretch variant: zero-sigma ParticlePrior 2D simplex barycenter interpolation | ParticlePrior (sigma=0) | COMPLETE / FAIL; 800/800 updates | atlas / unrecorded / 6e617abd2d29 | [definition](../toy_audit/api_contract/interpolation/publication.json); [readout](../toy_audit/api_contract/interpolation/publication.json); [recipe and provenance](../toy_audit/api_contract/interpolation/publication.json) |

### Experiment: interp-sparse-arc

Learn 12 zero-sigma ParticlePrior locations on a 270-degree 2D circular arc (R=2.0, sigma=0.03) without MoG smoothing so unseen latent k-NN chords remain on the curved manifold, cover >=22/32 arc bins, and progress smoothly without Voronoi snapping.

Explanation, interpretation and reproduction: [experiment readout](particle-interpolation-stretch/README.md).

Forge declarations: [interp_sparse_arc](../../configs/forge/tasks/interp_sparse_arc.json).

Declared Forge numerical gates and sampling:

[interp_sparse_arc](../../configs/forge/tasks/interp_sparse_arc.json)

| Metric | Required bound |
| --- | --- |
| finite_fraction | == 1 |
| endpoint_hq | >= 0.85 |
| interp_hq | >= 0.8 |
| interp_arc_bins | >= 22 |
| midpoint_novelty | >= 0.2 |

At least 5 consecutive passing terminal observations.
Execution guards: finite state = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | enumerated_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-interp-sparse-arc](../toy_audit/api_contract/interpolation/media/api-interp-sparse-arc.gif) | Learn 12 zero-sigma ParticlePrior locations on a 270-degree circular arc (R=2.0, sigma=0.03) so unseen latent k-NN chords stay on the curved arc, bridge >=22/32 angular bins, and avoid Voronoi snapping. Scope: Public API stretch variant: zero-sigma ParticlePrior latent chord interpolation | ParticlePrior (sigma=0) | COMPLETE / FAIL; 600/600 updates | atlas / unrecorded / 6e617abd2d29 | [definition](../toy_audit/api_contract/interpolation/publication.json); [readout](../toy_audit/api_contract/interpolation/publication.json); [recipe and provenance](../toy_audit/api_contract/interpolation/publication.json) |

### Experiment: mid-scale-identity

Checks identity preservation and target edit magnitude at intermediate control strength.

Forge declarations: [mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json), [mid_scale_identity_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mid_scale_identity_policy_cloud_coverage_v1.json), [recipe-prior-refactor-legacy-mid_scale_identity](../../configs/forge/tasks/recipe-prior-refactor-legacy-mid_scale_identity.json).

Declared Forge numerical gates and sampling:

[mid_scale_identity](../../configs/forge/tasks/mid_scale_identity.json)

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[mid_scale_identity_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mid_scale_identity_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_learned_parameter_measurement |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

[recipe-prior-refactor-legacy-mid_scale_identity](../../configs/forge/tasks/recipe-prior-refactor-legacy-mid_scale_identity.json)

| Metric | Required bound |
| --- | --- |
| concept_cos_plus | >= 0.85 |
| concept_cos_minus | >= 0.85 |
| concept_mag_plus | >= 0.75 |
| concept_mag_plus | <= 1.25 |
| concept_mag_minus | >= 0.75 |
| concept_mag_minus | <= 1.25 |
| identity_at_0 | >= 0.85 |
| identity_at_mid | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| mid_scale_identity | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| mid_scale_identity | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| mid_scale_identity | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| mid_scale_identity | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-midscale-identity](../toy_audit/api_contract/media/api-midscale-identity.gif) | Retain identity at half strength in addition to correct neutral and signed poles Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 800/800 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: misgan-block-neural-imputer

3-pair MisGAN grading neural imputer G_i([x*m, m, z_i]) on held-out ambiguous single-sensor queries without ppost.

Forge declarations: [misgan_block_neural_imputer](../../configs/forge/tasks/misgan_block_neural_imputer.json).

Declared Forge numerical gates and sampling:

[misgan_block_neural_imputer](../../configs/forge/tasks/misgan_block_neural_imputer.json)

| Metric | Required bound |
| --- | --- |
| observed_max_error | <= 1e-06 |
| acc | >= 0.95 |
| acc_lo | >= 0.65 |
| missing_std_ratio | >= 0.7 |
| missing_std_ratio | <= 1.3 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: misgan-block-recovery

2-pair incomplete-data MisGAN with detached mask gradient in L_x on 8D block sensor-dropout problem; recovers 2D manifold, 8D subspace, mask distribution and ppost Bayesian imputation.

Forge declarations: [misgan_block_recovery](../../configs/forge/tasks/misgan_block_recovery.json).

Declared Forge numerical gates and sampling:

[misgan_block_recovery](../../configs/forge/tasks/misgan_block_recovery.json)

| Metric | Required bound |
| --- | --- |
| modes | == 100 |
| hq | >= 0.9 |
| mass_tv | <= 0.1 |
| off_plane | <= 0.04 |
| mask_tv | <= 0.05 |
| acc | >= 0.96 |
| acc_lo | >= 0.7 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: misgan-mcar-p50-recovery

2-pair incomplete-data MisGAN on 8D MCAR p=0.5 coordinate-wise dropout (resampling empty masks); recovers 2D 100-mode manifold, 8D subspace, MCAR coordinate observation rate and ppost Bayesian imputation.

Forge declarations: [misgan_mcar_p50_recovery](../../configs/forge/tasks/misgan_mcar_p50_recovery.json).

Declared Forge numerical gates and sampling:

[misgan_mcar_p50_recovery](../../configs/forge/tasks/misgan_mcar_p50_recovery.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 90 |
| hq | >= 0.85 |
| mass_tv | <= 0.12 |
| off_plane | <= 0.05 |
| mask_tv | <= 0.05 |
| acc | >= 0.9 |
| impute_rmse | <= 0.12 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: misgan-mcar-p80-anchored

3-pair MisGAN on 8D extreme MCAR p=0.80 missingness with 10% complete anchor rows and neural imputer G_i trained with GAIN partial-hint conditioning on D_i([x_imp, h]).

Forge declarations: [misgan_mcar_p80_anchored](../../configs/forge/tasks/misgan_mcar_p80_anchored.json).

Declared Forge numerical gates and sampling:

[misgan_mcar_p80_anchored](../../configs/forge/tasks/misgan_mcar_p80_anchored.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 80 |
| hq | >= 0.8 |
| mass_tv | <= 0.15 |
| off_plane | <= 0.06 |
| imp_acc | >= 0.65 |
| imp_off_plane | <= 0.06 |
| impute_rmse | <= 0.18 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: mode-hold

Checks all eight ring modes and HQ through the sampled terminal hold; not within-mode density fidelity.

Forge declarations: [mode_hold](../../configs/forge/tasks/mode_hold.json), [mode_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/mode_hold_policy_cloud_coverage_as_served_v1.json), [mode_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mode_hold_policy_cloud_coverage_v1.json), [recipe-prior-refactor-legacy-mode_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-mode_hold.json), [recipe-prior-refactor-legacy-ring_extension](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_extension.json), [recipe-prior-refactor-legacy-ring_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json), [ring_extension](../../configs/forge/tasks/ring_extension.json), [ring_extension_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_extension_policy_cloud_coverage_as_served_v1.json), [ring_extension_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_extension_policy_cloud_coverage_v1.json), [ring_hold](../../configs/forge/tasks/ring_hold.json), [ring_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_hold_policy_cloud_coverage_as_served_v1.json), [ring_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_hold_policy_cloud_coverage_v1.json), [target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json), [target_shift_recovery_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/target_shift_recovery_policy_cloud_coverage_as_served_v1.json), [target_shift_recovery_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/target_shift_recovery_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[mode_hold](../../configs/forge/tasks/mode_hold.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[mode_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/mode_hold_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[mode_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/mode_hold_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[recipe-prior-refactor-legacy-mode_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-mode_hold.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 8 |
| hq | >= 0.9 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[recipe-prior-refactor-legacy-ring_extension](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_extension.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[recipe-prior-refactor-legacy-ring_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring_hold.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring_extension](../../configs/forge/tasks/ring_extension.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring_extension_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_extension_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[ring_extension_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_extension_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[ring_hold](../../configs/forge/tasks/ring_hold.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring_hold_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring_hold_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[ring_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring_hold_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| modes | == 8 |
| hq | >= 0.9 |
| hq | <= 1 |

Confirmation checks: 200.
Hold updates: 1200.
Extension updates: 300.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[target_shift_recovery](../../configs/forge/tasks/target_shift_recovery.json)

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[target_shift_recovery_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/target_shift_recovery_policy_cloud_coverage_as_served_v1.json)

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[target_shift_recovery_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/target_shift_recovery_policy_cloud_coverage_v1.json)

Recovery deadline in updates: 400.
Active quality must hold before the shift and throughout the post-deadline window; the matched frozen control must have zero passing post-deadline checks.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| mode_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| mode_hold | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| mode_hold | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| mode_hold | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |
| ring_extension | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| ring_hold | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ring8-acquire](../toy_audit/api_contract/media/api-ring8-acquire.gif) | Acquire all eight equal-weight radius-three, sigma-.07 Gaussian modes, including their within-mode law. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; hq >= 0.9, max_cov_eigen <= 1.5, max_radial_ks <= 0.1, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [api-ring8-hold](../toy_audit/api_contract/media/api-ring8-hold-failure.gif) | Acquire by update1200, then retain the same full eight-mode law without resetting optimizer state through update2400. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | ERROR / FAIL; 1200/2400 updates; API execution or metric error: RuntimeError: scientific prerequisite failed at update1200: ['hq >= 0.9', 'max_cov_eigen <= 1.5', 'max_radial_ks <= 0.1'] | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/failed-runs.json) |
| [api-ring8-resolution12](../toy_audit/api_contract/media/api-ring8-resolution12.gif) | Retain the original 12-row resource as a low-resource public-API control; assess the actual perturbed served law. A separate unperturbed twelve-equal-atom witness has a mass/width obstruction, which does not prove this stochastic served law impossible. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; hq >= 0.9, mass_tv <= 0.075, max_cov_eigen <= 1.5, max_radial_ks <= 0.1, min_cov_eigen >= 0.5, modes == 8, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |
| [api-ring8-shift](../toy_audit/api_contract/media/api-ring8-shift-failure.gif) | After qualified acquisition/hold, adapt to a +1 x translation at update2401; require recovery by update2800 and retained width/mass through3600. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | ERROR / FAIL; 1200/3600 updates; API execution or metric error: RuntimeError: scientific prerequisite failed at update1200: ['hq >= 0.9', 'max_cov_eigen <= 1.5', 'max_radial_ks <= 0.1'] | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/failed-runs.json) |

### Experiment: model-ae-gan-multimodal

Reconstruct four equally weighted noisy corner Gaussians and independently generate their modes, masses and within-mode shapes.

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/ae.md).

Forge declarations: [model_ae_gan_multimodal](../../configs/forge/tasks/model_ae_gan_multimodal.json).

Declared Forge numerical gates and sampling:

[model_ae_gan_multimodal](../../configs/forge/tasks/model_ae_gan_multimodal.json)

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.025 |
| reconstruction_mode_agreement | >= 0.98 |
| modes | >= 4 |
| precision | >= 0.9 |
| mass_tv | <= 0.15 |
| component_min_eigen_ratio | >= 0.1 |
| component_max_eigen_ratio | <= 4 |
| component_center_rms_sigma | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-ae-gan-multimodal](../toy_audit/api_contract/model_modes/media/api-model-ae-gan-multimodal.gif) | Reconstruct four equally weighted noisy corner Gaussians and independently generate their modes, masses and within-mode shapes. Scope: Four equally weighted Gaussians at (+/-1,+/-1), sigma=.08; independently drawn target and prior samples. Reconstruction and all prior gates must pass together. | MoGParticlePrior (sigma_rel=0) | COMPLETE / PASS; 1200/1200 updates; recomputed complete live curve and terminal suffix | bispf / cpu / 1e57b5216ea3 | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: model-ae-reconstruction

Reconstruct four equally weighted noisy corner Gaussians with public deterministic particle-AE routing and bounded offsets; explicit reconstruction-only objective control.

Explanation, interpretation and reproduction: [experiment readout](api-model-toys/ae.md).

Forge declarations: [model_ae_reconstruction](../../configs/forge/tasks/model_ae_reconstruction.json).

Declared Forge numerical gates and sampling:

[model_ae_reconstruction](../../configs/forge/tasks/model_ae_reconstruction.json)

| Metric | Required bound |
| --- | --- |
| reconstruction_mse | <= 0.025 |
| reconstruction_mode_agreement | >= 0.98 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, encoder, prior; exact optimizer updates = True; mechanism exercised = False.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-ae-reconstruction](../toy_audit/api_contract/model_modes/media/api-model-ae-reconstruction.gif) | Reconstruct four equally weighted noisy corner Gaussians with public deterministic particle-AE routing and bounded offsets; explicit reconstruction-only objective control. Scope: Four equally weighted Gaussians at (+/-1,+/-1), sigma=.08; independently drawn target and prior samples. Explicit reconstruction-only objective control; independent generation is measured but not required. | MoGParticlePrior (sigma_rel=0) | COMPLETE / PASS; 1200/1200 updates; recomputed complete live curve and terminal suffix | bispf / cpu / 1e57b5216ea3 | [definition](../toy_audit/api_contract/model_modes/publication.json); [readout](../toy_audit/api_contract/model_modes/publication.json); [recipe and provenance](../toy_audit/api_contract/model_modes/publication.json) |

### Experiment: model-misgan-four-mode

Learn an 8D four-mode complete distribution and the block-mask law using only incomplete rows, then learn a stochastic neural imputer whose held-out single-sensor queries cover both ambiguous branches with correct mass.

Explanation, interpretation and reproduction: [experiment readout](misgan-bispf/README.md).

Forge declarations: [model_misgan_four_mode](../../configs/forge/tasks/model_misgan_four_mode.json).

Declared Forge numerical gates and sampling:

[model_misgan_four_mode](../../configs/forge/tasks/model_misgan_four_mode.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 4 |
| hq | >= 0.9 |
| mass_tv | <= 0.1 |
| off_plane | <= 0.04 |
| mask_tv | <= 0.05 |
| imputer_observed_error | <= 1e-06 |
| imputer_accuracy | >= 0.94 |
| ambiguous_hq | >= 0.9 |
| ambiguous_mass_tv | <= 0.12 |
| ambiguous_query_coverage | >= 0.9 |
| ambiguous_std_ratio | >= 0.7 |
| ambiguous_std_ratio | <= 1.3 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, prior, discriminator; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | task_model_kernel_and_posterior_without_serving_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-model-misgan-four-mode](../toy_audit/api_contract/misgan_bispf/media/bispf-misgan-transport-settle-v1.gif) | Learn an 8D four-mode complete distribution and the block-mask law using only incomplete rows, then learn a stochastic neural imputer whose held-out single-sensor queries cover both ambiguous branches with correct mass. Scope: Explicit four-corner sigma=.08 target with a fixed Hadamard 8x2 lift. Same four block masks and masses as original MisGAN; sensor7 sees x+y, leaving two equally probable branches at zero. Original 100-mode MLP questions remain unchanged. Absolute task-owned MoG sigma=.025; relative calibration is bypassed. Recommended scoped recipe: bispf-misgan-transport-settle-v1 | MoGParticlePrior (sigma_rel=0) | COMPLETE / PASS; 1200/1200 updates | bispf / cuda:0 / d2b5cf8eba1f | [definition](../toy_audit/api_contract/misgan_bispf/publication.json); [readout](../toy_audit/api_contract/misgan_bispf/publication.json); [recipe and provenance](../toy_audit/api_contract/misgan_bispf/publication.json) |

### Experiment: pacgan8-grid100

PacGAN m=8 packed discriminator on 100-mode 10x10 2D Gaussian grid (centers in {-9,-7,...,7,9}^2, sigma=0.05) testing high-order product-measure mode coverage and within-mode covariance preservation.

Forge declarations: [pacgan8_grid100](../../configs/forge/tasks/pacgan8_grid100.json).

Declared Forge numerical gates and sampling:

[pacgan8_grid100](../../configs/forge/tasks/pacgan8_grid100.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 95 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| min_cov_eig_ratio | >= 0.15 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.05) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: pacgan-unequal-mass

PacGAN m=4 packed discriminator on 5-mode 2D Gaussian ring (R=2.0, sigma=0.08) with unequal mode weights (0.40, 0.30, 0.20, 0.08, 0.02) testing rare 2% mode survival without within-mode covariance collapse.

Forge declarations: [pacgan_unequal_mass](../../configs/forge/tasks/pacgan_unequal_mass.json).

Declared Forge numerical gates and sampling:

[pacgan_unequal_mass](../../configs/forge/tasks/pacgan_unequal_mass.json)

| Metric | Required bound |
| --- | --- |
| modes | == 5 |
| min_mode_mass | >= 0.01 |
| mass_tv | <= 0.1 |
| hq | >= 0.88 |
| min_cov_eig_ratio | >= 0.2 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.05) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: precision-gaussian

Train N(2, 0.5^2) from scratch using FP32 as the matched full-precision control.

Forge declarations: [precision_gaussian_fp32](../../configs/forge/tasks/precision_gaussian_fp32.json), [precision_gaussian_mxfp8](../../configs/forge/tasks/precision_gaussian_mxfp8.json), [precision_gaussian_nvfp4](../../configs/forge/tasks/precision_gaussian_nvfp4.json).

Declared Forge numerical gates and sampling:

[precision_gaussian_fp32](../../configs/forge/tasks/precision_gaussian_fp32.json), [precision_gaussian_mxfp8](../../configs/forge/tasks/precision_gaussian_mxfp8.json), [precision_gaussian_nvfp4](../../configs/forge/tasks/precision_gaussian_nvfp4.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| finite_fraction | == 1 |
| mean_error_sigma | <= 0.2 |
| std_ratio | >= 0.8 |
| std_ratio | <= 1.2 |
| cdf_ks | <= 0.05 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; rng isolation = True; optimizer roles = generator, discriminator, prior; exact optimizer updates = True; mechanism exercised = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-gaussian1d-acquisition](../toy_audit/api_contract/gaussian1d/goal.gif) | Can the public ParticleGAN trainer acquire the scalar law N(2, 0.5^2) from random initialization within 1,000 updates, with correct location, width and CDF shape at five terminal checks? Scope: Acquire N(2, .5^2) in 1,000 updates from random initialization; exact Gaussian CDF, location and width gates must pass at five terminal checks. Tier 1 remains provisional; this standalone API cohort grants no whole-view qualification. | MoGParticlePrior (sigma_rel=0) | COMPLETE / FAIL; 1000/1000 updates; cdf_ks <= 0.05, last 5 post-update metric observations do not all pass | k3p / cpu / 666642486d3c | [definition](../toy_audit/api_contract/gaussian1d/results.json); [readout](../toy_audit/api_contract/gaussian1d/results.json); [recipe and provenance](../toy_audit/api_contract/gaussian1d/results.json) |

### Experiment: recipe-prior-refactor-legacy-clockfree-audit-measurement-v1

Measure the original four clock/state parity conditions and zero-dependency condition for an explicitly scheduled recipe; diagnostic FAIL grants no clock-free claim or qualification.

Forge declarations: [recipe-prior-refactor-legacy-clockfree_audit_measurement_v1](../../configs/forge/tasks/recipe-prior-refactor-legacy-clockfree_audit_measurement_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-clockfree_audit_measurement_v1](../../configs/forge/tasks/recipe-prior-refactor-legacy-clockfree_audit_measurement_v1.json)

Exact state/output parity for: step_label, horizon, evaluation_cadence, restart; bound source audit required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: residual-student

Checks whether the intended residual moves toward the correct paired target.

Forge declarations: [recipe-prior-refactor-legacy-residual_student](../../configs/forge/tasks/recipe-prior-refactor-legacy-residual_student.json), [residual_student](../../configs/forge/tasks/residual_student.json), [residual_student_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/residual_student_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-residual_student](../../configs/forge/tasks/recipe-prior-refactor-legacy-residual_student.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

[residual_student](../../configs/forge/tasks/residual_student.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

[residual_student_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/residual_student_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |
| success_rate | >= 1 |
| wrong_pad_rate | <= 0 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | public_recipe_schedule |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| residual_student | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| residual_student | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| residual_student | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| residual_student | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-trajectory-residual](../toy_audit/api_contract/media/api-trajectory-residual.gif) | Recover the fast trajectory with a residual head and reject a correct marginal with wrong identities Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: ring16-acquisition

Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread within 1600 updates. No extended hold phase.

Forge declarations: [recipe-prior-refactor-legacy-ring16_acquisition](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json), [ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json), [ring16_acquisition_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring16_acquisition_policy_cloud_coverage_as_served_v1.json), [ring16_acquisition_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring16_acquisition_policy_cloud_coverage_v1.json), [ring16_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-ring16_acquisition](../../configs/forge/tasks/recipe-prior-refactor-legacy-ring16_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring16_acquisition](../../configs/forge/tasks/ring16_acquisition.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[ring16_acquisition_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/ring16_acquisition_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[ring16_acquisition_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/ring16_acquisition_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 96 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[ring16_acquisition_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/ring16_acquisition_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| sample_count | >= 4096 |
| modes | >= 16 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ring16_acquisition | [Atlas](../../configs/forge/ideas/atlas-particle-normal-v4.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / 6261a5458c1f / 767710fe7bd0 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [K3P](../../configs/forge/ideas/recipe-prior-refactor-k3p-v1.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 3c8a0c05c790 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [KA2](../../configs/forge/ideas/recipe-prior-refactor-ka2-v1.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / fb4738652dc8 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [R1/R2](../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / a266b096fb60 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [GAN v3 release 0.7](../../configs/forge/ideas/recipe-prior-refactor-release07-gan-v3-v1.json) | MoGParticlePrior (sigma=0.1) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 1accaa098064 | [source-bound receipt index](technique-inventory.json) |
| ring16_acquisition | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.1) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-ring16-acquisition](../toy_audit/api_contract/ring16/goal.gif) | Acquire all 16 equally weighted two-dimensional Gaussian clusters from scratch: radius 3, sigma 0.1; require meaningful occupancy in every cluster, roughly balanced mass and noncollapsed local spread within 400 updates. No extended hold phase. Scope: From random initialization, acquire sixteen equal radius-three sigma-.1 Gaussian clusters within 400 updates. Five terminal acquisition checks add no hold phase. Tier 1 is provisional; this standalone K3P/API initializer and RNG cohort supplies no Forge promotion credit. | MoGParticlePrior (sigma_rel=0) | ERROR / FAIL; 400/400 updates; component_covariance_error <= 0.85, hq >= 0.85, last 5 post-update metric observations do not all pass, goal media/state error: ModuleNotFoundError: No module named 'matplotlib' | k3p / cuda:0 / 746146cbdc5a | [definition](../toy_audit/api_contract/ring16/publication.json); [readout](../toy_audit/api_contract/ring16/publication.json); [recipe and provenance](../toy_audit/api_contract/ring16/publication.json) |

### Experiment: ring16-bigan-continuous

Continuous 2D multimodal joint (G(z), E(x), D(x, z)) on 16-Gaussian ring with MoGParticlePrior(sigma=0.1).

Forge declarations: [ring16_bigan_continuous](../../configs/forge/tasks/ring16_bigan_continuous.json).

Declared Forge numerical gates and sampling:

[ring16_bigan_continuous](../../configs/forge/tasks/ring16_bigan_continuous.json)

| Metric | Required bound |
| --- | --- |
| modes | >= 16 |
| hq | >= 0.85 |
| mass_tv | <= 0.15 |
| recon_rmse | <= 0.12 |
| mode_consistency | >= 0.98 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: rotated100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [recipe-prior-refactor-legacy-rotated100](../../configs/forge/tasks/recipe-prior-refactor-legacy-rotated100.json), [rotated100](../../configs/forge/tasks/rotated100.json), [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json), [rotated100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_14k_policy_cloud_coverage_as_served_v1.json), [rotated100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_14k_policy_cloud_coverage_v1.json), [rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json), [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json), [rotated100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_policy_cloud_coverage_as_served_v1.json), [rotated100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-rotated100](../../configs/forge/tasks/recipe-prior-refactor-legacy-rotated100.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[rotated100](../../configs/forge/tasks/rotated100.json), [rotated100_14k](../../configs/forge/tasks/rotated100_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[rotated100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_14k_policy_cloud_coverage_as_served_v1.json), [rotated100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/rotated100_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[rotated100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_14k_policy_cloud_coverage_v1.json), [rotated100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/rotated100_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[rotated100_affine_square_named_v1](../../configs/forge/tasks/rotated100_affine_square_named_v1.json), [rotated100_affine_square_named_v1_14k](../../configs/forge/tasks/rotated100_affine_square_named_v1_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| rotated100 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| rotated100 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| rotated100 | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| rotated100 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-rotated100](../toy_audit/api_contract/media/api-rotated100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: staggered100

Recover all 100 Gaussian components, balanced mass, centers and within-mode covariance/radial spread; distinguish clean from noisy served laws.

Forge declarations: [recipe-prior-refactor-legacy-staggered100](../../configs/forge/tasks/recipe-prior-refactor-legacy-staggered100.json), [staggered100](../../configs/forge/tasks/staggered100.json), [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json), [staggered100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_14k_policy_cloud_coverage_as_served_v1.json), [staggered100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_14k_policy_cloud_coverage_v1.json), [staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json), [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json), [staggered100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_policy_cloud_coverage_as_served_v1.json), [staggered100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-staggered100](../../configs/forge/tasks/recipe-prior-refactor-legacy-staggered100.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[staggered100](../../configs/forge/tasks/staggered100.json), [staggered100_14k](../../configs/forge/tasks/staggered100_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[staggered100_14k_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_14k_policy_cloud_coverage_as_served_v1.json), [staggered100_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/staggered100_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[staggered100_14k_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_14k_policy_cloud_coverage_v1.json), [staggered100_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/staggered100_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| coverage.all_finite | == True |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_modes | >= 100 |
| coverage.min_precision | >= 0.97 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.min_samples | >= 20000 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.mass_tv | <= 0.06 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

[staggered100_affine_square_named_v1](../../configs/forge/tasks/staggered100_affine_square_named_v1.json), [staggered100_affine_square_named_v1_14k](../../configs/forge/tasks/staggered100_affine_square_named_v1_14k.json)

| Metric | Required bound |
| --- | --- |
| coverage.min_samples | >= 20000 |
| coverage.min_modes | >= 100 |
| coverage.min_hq_mode_mass | >= 0.005 |
| coverage.min_precision | >= 0.75 |
| coverage.max_mass_tv | <= 0.1 |
| coverage.max_mode_mass | <= 0.02 |
| coverage.min_cov_eig_ratio | >= 0.4 |
| coverage.max_cov_eig_ratio | <= 1.7 |
| coverage.min_radial_median_ratio | >= 0.65 |
| coverage.max_radial_median_ratio | <= 1.4 |
| coverage.all_finite | == True |
| accuracy.mass_tv | <= 0.06 |
| accuracy.center_rms_sigma | <= 0.2 |
| accuracy.abs_cov_trace_bias | <= 0.1 |
| accuracy.radial_ks | <= 0.04 |

At least 5 consecutive passing terminal observations.
Both sustained coverage and independent holdout accuracy must pass.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| staggered100 | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| staggered100 | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| staggered100 | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| staggered100 | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-staggered100](../toy_audit/api_contract/media/api-staggered100.gif) | Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / PASS; 7000/7000 updates | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: trajectory

Checks the extracted trajectory edit while preserving identity in finite paired rows.

Forge declarations: [recipe-prior-refactor-legacy-trajectory](../../configs/forge/tasks/recipe-prior-refactor-legacy-trajectory.json), [trajectory](../../configs/forge/tasks/trajectory.json), [trajectory_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/trajectory_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-trajectory](../../configs/forge/tasks/recipe-prior-refactor-legacy-trajectory.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

[trajectory](../../configs/forge/tasks/trajectory.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | live |
| Evaluation output noise | public_recipe_schedule |

[trajectory_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/trajectory_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| identity_mse | <= 0.02 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_conditional_prior_centers_with_scheduled_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | public_recipe_schedule |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| trajectory | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| trajectory | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| trajectory | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| trajectory | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-trajectory-edit](../toy_audit/api_contract/media/api-trajectory-edit.gif) | Change angular speed while preserving each trajectory's radius and starting phase Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: two-pole

Does an unchanged global recipe move from zero by 800 updates with schedule horizon 800? Movement and bounded slope are the declared question; two-mode fidelity is diagnostic only.

Explanation, interpretation and reproduction: [experiment readout](k3p-two-pole-horizon-v1/README.md).

Forge declarations: [recipe-prior-refactor-legacy-two_pole](../../configs/forge/tasks/recipe-prior-refactor-legacy-two_pole.json), [two_pole](../../configs/forge/tasks/two_pole.json), [two_pole_800_schedule800_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json), [two_pole_800_schedule80_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json), [two_pole_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/two_pole_policy_cloud_coverage_v1.json), [two_pole_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-two_pole](../../configs/forge/tasks/recipe-prior-refactor-legacy-two_pole.json), [two_pole_800_schedule800_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule800_diagnostic_v1.json), [two_pole_800_schedule80_diagnostic_v1](../../configs/forge/tasks/two_pole_800_schedule80_diagnostic_v1.json)

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_particles_and_critic_gradient |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[two_pole](../../configs/forge/tasks/two_pole.json)

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = prior, discriminator; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_particles_and_critic_gradient |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[two_pole_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/two_pole_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_learned_particles_and_critic_gradient |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

[two_pole_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/two_pole_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| mean_abs | >= 0.3 |
| grad_med | <= 1 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = prior, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_learned_particles_and_critic_gradient |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| two_pole | [Atlas](../../configs/forge/ideas/atlas-particle-normal-v4.json) | ParticlePrior (sigma=0) | FAIL | CHANGED; earlier contract | cuda / 6261a5458c1f / 767710fe7bd0 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| two_pole | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| two_pole | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| two_pole | [K3P](../../configs/forge/ideas/recipe-prior-refactor-k3p-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 3c8a0c05c790 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [KA2](../../configs/forge/ideas/recipe-prior-refactor-ka2-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / fb4738652dc8 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [R1/R2](../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / a266b096fb60 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [GAN v3 release 0.7](../../configs/forge/ideas/recipe-prior-refactor-release07-gan-v3-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 1accaa098064 | [source-bound receipt index](technique-inventory.json) |
| two_pole | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-two-pole-grid12](../toy_audit/api_contract/media/api-two-pole-grid12.gif) | Check six target offsets per pole in one full row-ID realization, beyond mere travel; full served output-law fidelity remains unmeasured when latent perturbation is active. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 80/80 updates; max_grid_quantile_error_halfwidth <= 0.1, support_fraction >= 0.95, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: unipolar

Checks an intended edit with preservation of unrelated content.

Forge declarations: [recipe-prior-refactor-legacy-unipolar](../../configs/forge/tasks/recipe-prior-refactor-legacy-unipolar.json), [unipolar](../../configs/forge/tasks/unipolar.json), [unipolar_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unipolar_policy_cloud_coverage_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-unipolar](../../configs/forge/tasks/recipe-prior-refactor-legacy-unipolar.json)

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[unipolar](../../configs/forge/tasks/unipolar.json)

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[unipolar_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unipolar_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| cover | >= 0.85 |
| off_caption | <= 0.05 |
| neu_hold | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_learned_parameter_measurement |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| unipolar | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| unipolar | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| unipolar | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| unipolar | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-unipolar-hold](../toy_audit/api_contract/media/api-unipolar-hold.gif) | Make the positive 4D edit while holding the free scale-zero origin Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / PASS; 400/400 updates | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: unused-token-hold

Checks that active controls move and unused controls remain unchanged.

Forge declarations: [recipe-prior-refactor-legacy-unused_token_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-unused_token_hold.json), [unused_token_hold](../../configs/forge/tasks/unused_token_hold.json), [unused_token_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unused_token_hold_policy_cloud_coverage_v1.json), [unused_token_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-unused_token_hold](../../configs/forge/tasks/recipe-prior-refactor-legacy-unused_token_hold.json)

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[unused_token_hold](../../configs/forge/tasks/unused_token_hold.json)

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; optimizer roles = generator, discriminator; mechanism exercised = True; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | learned_parameter_measurement |
| Scoring weights | live |
| Evaluation output noise | not_applied_to_measurement |

[unused_token_hold_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/unused_token_hold_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_learned_parameter_measurement |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

[unused_token_hold_tier1_policy_selected_cloud_v1](../../configs/forge/task-variants/tier1_policy_selected_cloud_v1/unused_token_hold_tier1_policy_selected_cloud_v1.json)

| Metric | Required bound |
| --- | --- |
| unused_hold | >= 0.85 |
| concept_move | >= 0.85 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.
Execution guards: finite state = True; mechanism exercised = True; optimizer roles = generator, discriminator; rng isolation = True.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | tier1_selected_learned_parameter_measurement |
| Scoring weights | state_selected |
| Evaluation output noise | not_applied_to_measurement |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| unused_token_hold | [Atlas](../../configs/forge/ideas/atlas-particle-normal-v4.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / 6261a5458c1f / 767710fe7bd0 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [EMPF](../../configs/forge/ideas/empf-v7.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [K3P](../../configs/forge/ideas/recipe-prior-refactor-k3p-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 3c8a0c05c790 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [KA2](../../configs/forge/ideas/recipe-prior-refactor-ka2-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / fb4738652dc8 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [R1/R2](../../configs/forge/ideas/recipe-prior-refactor-r1r2-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / a266b096fb60 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [GAN v3 release 0.7](../../configs/forge/ideas/recipe-prior-refactor-release07-gan-v3-v1.json) | ParticlePrior (sigma=0) | PASS | CHANGED; earlier contract | cuda / a0493ca3c213 / 1accaa098064 | [source-bound receipt index](technique-inventory.json) |
| unused_token_hold | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | ParticlePrior (sigma=0) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-unused-token-hold](../toy_audit/api_contract/media/api-unused-token-hold.gif) | Move the concept slot on its target axis while keeping the unused slot fixed Scope: New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged. | MoGParticlePrior (sigma_rel=0.25) | COMPLETE / FAIL; 200/200 updates; concept_move, last 5 post-update metric observations do not all pass | conditional_ka2 / cpu / 58bcb6febe12 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-anisotropic

Checks covariance shape: a narrow axis cannot be rescued by a wide one.

Forge declarations: [recipe-prior-refactor-legacy-vector_anisotropic](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_anisotropic.json), [vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json), [vector_anisotropic_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_anisotropic_policy_cloud_coverage_as_served_v1.json), [vector_anisotropic_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_anisotropic_policy_cloud_coverage_v1.json), [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-vector_anisotropic](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_anisotropic.json), [vector_anisotropic_published](../../configs/forge/tasks/vector_anisotropic_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_anisotropic](../../configs/forge/tasks/vector_anisotropic.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_anisotropic_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_anisotropic_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[vector_anisotropic_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_anisotropic_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_anisotropic | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| vector_anisotropic | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| vector_anisotropic | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| vector_anisotropic | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-anisotropic](../toy_audit/api_contract/media/api-vector-anisotropic.gif) | Checks covariance shape: a narrow axis cannot be rescued by a wide one. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, resolved_max_component_spill <= 0.05, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-overlap

Scores the observable distribution when latent components are not identifiable.

Forge declarations: [recipe-prior-refactor-legacy-vector_overlap](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_overlap.json), [vector_overlap](../../configs/forge/tasks/vector_overlap.json), [vector_overlap_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_overlap_policy_cloud_coverage_as_served_v1.json), [vector_overlap_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_overlap_policy_cloud_coverage_v1.json), [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-vector_overlap](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_overlap.json), [vector_overlap_published](../../configs/forge/tasks/vector_overlap_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_overlap](../../configs/forge/tasks/vector_overlap.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_overlap_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_overlap_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[vector_overlap_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_overlap_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_overlap | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| vector_overlap | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| vector_overlap | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| vector_overlap | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-overlap](../toy_audit/api_contract/media/api-vector-overlap.gif) | Scores the observable distribution when latent components are not identifiable. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-spiral

Checks continuous curved mass rather than a finite list of target mode centers.

Forge declarations: [recipe-prior-refactor-legacy-vector_spiral](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_spiral.json), [vector_spiral](../../configs/forge/tasks/vector_spiral.json), [vector_spiral_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_spiral_policy_cloud_coverage_as_served_v1.json), [vector_spiral_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_spiral_policy_cloud_coverage_v1.json), [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-vector_spiral](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_spiral.json), [vector_spiral_published](../../configs/forge/tasks/vector_spiral_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_spiral](../../configs/forge/tasks/vector_spiral.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_spiral_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_spiral_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[vector_spiral_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_spiral_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mean_error | <= 0.15 |
| covariance_error | <= 0.45 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_spiral | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| vector_spiral | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| vector_spiral | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| vector_spiral | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-spiral](../toy_audit/api_contract/media/api-vector-spiral.gif) | Checks continuous curved mass rather than a finite list of target mode centers. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1600/1600 updates; last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-two-broad

Basic learnable multimodal distribution and within-mode spread.

Forge declarations: [recipe-prior-refactor-legacy-vector_two_broad](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_two_broad.json), [vector_two_broad](../../configs/forge/tasks/vector_two_broad.json), [vector_two_broad_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_two_broad_policy_cloud_coverage_as_served_v1.json), [vector_two_broad_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_two_broad_policy_cloud_coverage_v1.json), [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-vector_two_broad](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_two_broad.json), [vector_two_broad_published](../../configs/forge/tasks/vector_two_broad_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_two_broad](../../configs/forge/tasks/vector_two_broad.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_two_broad_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_two_broad_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[vector_two_broad_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_two_broad_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_two_broad | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| vector_two_broad | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| vector_two_broad | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| vector_two_broad | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | PASS | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-two-broad](../toy_audit/api_contract/media/api-vector-two-broad.gif) | Basic learnable multimodal distribution and within-mode spread. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-unequal-mass

Checks target occupancy including the rare 2% component, not uniformity.

Forge declarations: [recipe-prior-refactor-legacy-vector_unequal_mass](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_unequal_mass.json), [vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json), [vector_unequal_mass_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_unequal_mass_policy_cloud_coverage_as_served_v1.json), [vector_unequal_mass_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_unequal_mass_policy_cloud_coverage_v1.json), [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-vector_unequal_mass](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_unequal_mass.json), [vector_unequal_mass_published](../../configs/forge/tasks/vector_unequal_mass_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_unequal_mass](../../configs/forge/tasks/vector_unequal_mass.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_unequal_mass_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_unequal_mass_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[vector_unequal_mass_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_unequal_mass_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |
| min_mass_ratio | >= 0.25 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_unequal_mass | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| vector_unequal_mass | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| vector_unequal_mass | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| vector_unequal_mass | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-unequal-mass](../toy_audit/api_contract/media/api-vector-unequal-mass.gif) | Checks target occupancy including the rare 2% component, not uniformity. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: vector-unequal-width

Checks component-specific scales without imposing one shared Gaussian width.

Forge declarations: [recipe-prior-refactor-legacy-vector_unequal_width](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_unequal_width.json), [vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json), [vector_unequal_width_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_unequal_width_policy_cloud_coverage_as_served_v1.json), [vector_unequal_width_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_unequal_width_policy_cloud_coverage_v1.json), [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json).

Declared Forge numerical gates and sampling:

[recipe-prior-refactor-legacy-vector_unequal_width](../../configs/forge/tasks/recipe-prior-refactor-legacy-vector_unequal_width.json), [vector_unequal_width_published](../../configs/forge/tasks/vector_unequal_width_published.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_unequal_width](../../configs/forge/tasks/vector_unequal_width.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.025) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | live |
| Evaluation output noise | clean |

[vector_unequal_width_policy_cloud_coverage_as_served_v1](../../configs/forge/task-variants/policy_cloud_coverage_as_served_v1/vector_unequal_width_policy_cloud_coverage_as_served_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_as_served_public_prior_with_learned_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | learned_as_served |

[vector_unequal_width_policy_cloud_coverage_v1](../../configs/forge/task-variants/policy_cloud_coverage_v1/vector_unequal_width_policy_cloud_coverage_v1.json)

| Metric | Required bound |
| --- | --- |
| sw1_normalized | <= 0.18 |
| mass_tv | <= 0.15 |
| hq | >= 0.85 |
| component_covariance_error | <= 0.85 |
| component_min_eigen_ratio | >= 0.15 |

At least 5 consecutive passing terminal observations.
All 24 declared observations and final live metrics are required.

| Measurement | Declared condition |
| --- | --- |
| Prior | ParticlePrior (sigma=0) |
| Sampling law | policy_cloud_public_prior_without_output_noise |
| Scoring weights | state_selected |
| Evaluation output noise | clean |

Recorded Forge task outcomes (exact saved configuration/source/runtime):

| Task | Configuration | Recorded prior code path | Recorded outcome | Current declaration | Source / cohort | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| vector_unequal_width | [BCAP dualnorm (experimental starting point)](../../configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / a0493ca3c213 / 8a6ee0b1c9ce | [source-bound receipt index](technique-inventory.json) |
| vector_unequal_width | [BISPF](../../configs/forge/ideas/bispf-optimized-quintic-v1.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / d9d72d53761c / a5c1a7106c9f | [source-bound receipt index](technique-inventory.json) |
| vector_unequal_width | [EMPF](../../configs/forge/ideas/empf-v7.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 3347cd63c16c / 6bb0a70f936f | [source-bound receipt index](technique-inventory.json) |
| vector_unequal_width | [RHBI](../../configs/forge/ideas/rhbi-v11.json) | MoGParticlePrior (sigma=0.025) | FAIL | matches; source remains frozen | cuda / 5a6af3846559 / 5d64d3269823 | [source-bound receipt index](technique-inventory.json) |

Related public-API demonstrations, with their own recorded contracts:

| Variant / actual-training GIF | What this variant tests | Recorded prior code path | Recorded result / failed bounds | Recipe / compute / source | Evidence |
| --- | --- | --- | --- | --- | --- |
| [api-vector-unequal-width](../toy_audit/api_contract/media/api-vector-unequal-width.gif) | Checks component-specific scales without imposing one shared Gaussian width. Scope: A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced. | ParticlePrior (sigma=0) | COMPLETE / FAIL; 1200/1200 updates; projection_ks <= 0.06, resolved_max_component_spill <= 0.05, last 5 post-update metric observations do not all pass | atlas / cpu / d9d51eff83a2 | [definition](../toy_audit/api_contract/cases.json); [readout](../toy_audit/api_contract/readout.json); [recipe and provenance](../toy_audit/api_contract/runs.json) |

### Experiment: zero-tanh-rail-escape

Execute the declared zero_tanh_rail_escape gate.

Forge declarations: [zero_tanh_rail_escape](../../configs/forge/tasks/zero_tanh_rail_escape.json).

Declared Forge numerical gates and sampling:

[zero_tanh_rail_escape](../../configs/forge/tasks/zero_tanh_rail_escape.json)

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |

Recovery deadline in updates: 400.

| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

### Experiment: zero-tanh-smoke

Execute the declared zero_tanh_smoke gate.

Forge declarations: [zero_tanh_smoke](../../configs/forge/tasks/zero_tanh_smoke.json).

Declared Forge numerical gates and sampling:

[zero_tanh_smoke](../../configs/forge/tasks/zero_tanh_smoke.json)

| Metric | Required bound |
| --- | --- |
| mean_abs_error | <= 0.005 |
| rms_error | <= 0.01 |
| p99_abs_error | <= 0.025 |
| saturated_fraction | == 0 |


| Measurement | Declared condition |
| --- | --- |
| Prior | MoGParticlePrior (sigma=0.1) |
| Sampling law | public_prior_without_output_noise |
| Scoring weights | — |
| Evaluation output noise | clean |

No measured Forge outcome for these exact task IDs in the current solution publication. Consult the solution leaderboard for unknown requirements and capability blockers.

No related published API training GIF. This task retains its own declared numerical audit.

## Keep this view current

After editing task/view declarations or publishing new compact results and media indexes, run `python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md` and commit this same report. New tasks and explicit API question mappings are discovered automatically; no training, rescoring or queue access is needed. The report freshness test detects stale generated content.

The wider question review also links standalone experiments outside the Forge tier catalog. Adding a diagnostic there does not assign it to a Forge tier.

- [Questions](../toy_audit/api_contract/QUESTION_RANKING.md)
- [Later questions](../toy_audit/api_contract/recent_prs/README.md)
- [Caption questions](../toy_audit/api_contract/caption_prs/README.md)

Declaration input digest: `10c21db77eab281956e145946d5ec1f051ba12f2eba231414ead58b6843f5737`. The JSON form includes the individual task and view file hashes.

Published artifact input digest: `776baa6cf098bb76a284b850cb3d2cec1ae521ca1acb11ffd90e3344477cce31`. Artifact hashes and exact recipe/source/runtime bindings are included in the JSON form.
