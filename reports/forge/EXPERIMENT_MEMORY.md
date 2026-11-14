# ParticleGAN Forge experiment memory

Read the relevant prior evidence before declaring an idea. Historical outcomes retain their original scope. Published qualification and automation snapshots are retained without replaying unavailable originals. Compact publications enter recall as display-only summaries and grant no qualification.

Records: 997. Inventory coverage: complete. Unresolved import items: 7.

## Goal views

- [adaptation](leaderboards/adaptation.md)
- [clockfree_continuous](leaderboards/clockfree_continuous.md)
- [discriminator_stability](technique-inventory.md)
- [formulation_comparison](leaderboards/formulation_comparison.md)
- [host_profile_transfer](leaderboards/host_profile_transfer.md)
- [policy_cloud_coverage](technique-inventory.md)
- [policy_cloud_coverage_as_served](technique-inventory.md)
- [quality_coverage](leaderboards/quality_coverage.md)
- [tier1_policy_coverage](technique-inventory.md)

[Measured automation costs, reuse, and avoided work](automation.json). Run `python -m experiments.forge stats` for current accounting; unavailable measurements remain explicit.

## Pending readouts

bcap--148a58dc4036cfc2d71a10b427582aeea304b4ac70874ff2fe7ec02f31cee561, bcap--17c9d255a2f8e1adfc1e4ec686e3cee3c7e3fd40fc7979328d4e1955cc2d29a1, bcap--263d190ebb4c40ddb21337a16a598f5ffcca4639031b3b70aaa3036aec67b31e, bcap--655141ae1d99b0b08dcbac36abbb6f135473d47d389f2ec9d41d9aa6d1201fbe, bcap--6c72e4aaf8ffe22be663a6e041971cc22bbee9402bb0095c4935aa167b6ccb03, bcap--8a9884cad1e87d5d950d8dff3d4901a787e8f94fc99b00e83877fdf6e1a932c2, bcap--90dcc2c3039f9f4c7ed2b1de03dec0c75cec2e9d386de0bb302f90a5f923de23, bcap--a3675bf5a89a136bd4c64b5577021215c1e118aca2971e274b6d23ded6ac55a9, bcap--c89e202b7830cfc38743c25fc1b34ca8ceef3f113c4f091017cac7c843972da3, bcap--ccfa83125643932bdfc88a2178a94ec019a2687d7915dae1ed0d6852925f603d, bcap--e8c9fcd62a13b748b0a1c97543b828ee618833ea45789681323a37c9c300d773, bcap--f0163dead110803a0e982cacfa2589918cd7345eea1655c76a3e3f21b0037887, bcap-original-horizon-diagnostics-v1, bcap-tier2-round6-control-v1, bcap-tier2-round6-half-network-v1, bcap-tier2-round6-late-anneal-v1, bcap-tier2-round6-prior-smoothing-v1, bcap-tier2-round6-zero-damping-v1, constraint_geometry-control-v1, constraint_geometry-nonascent-v1, five-word-joint-ka2-v1, hydraulic-output-travel-v1, information_geometry_spectral_half_v1, kinetic_transport_sliced_v1, thermodynamic-bcap-optimism-v1

## Experiment and family records

### api-model-ddgan-failure-investigation · api-model-ddgan-failure-investigation-v1

**Scope:** historical; family_context; revision `e1a3a9e812dc568ed94e9774427708899f4469695d2415d049ee8ab423488b52`.

Distinguish errors in low-noise conditional generation, unusual fixed contexts, high-noise kernel masking and the saved critic local correction field.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `saved_state_failure_analysis`.



Both saved Gaussian chains have low-noise conditional mismatches at central and fresh marginal contexts (step-1 median CDF errors 0.265 DDGAN, 0.301 AE-DDGAN). At step 4, analytic reverse noise supplies about 99 percent of transition variance; vanilla clean CDF error 0.271 can coexist with reverse CDF error 0.033. Local per-example output gradients oppose mean correction on 7/8 vanilla and 4/8 composed fixed panels; this does not identify parameter-step direction or a temporal cause. Original generation FAILs, paired reconstruction subgate PASS and blocked deeper runs remain unchanged.

**Next:** Propose a separate bounded stationary-generator critic-fit diagnostic before changing a trainer. Preserve one global recipe, original gates and all failures; no new training, continuation, seed sweep or default promotion in this investigation.

[Evidence](../../reports/forge/ddgan-failure-investigation/README.md) · [Record](records/api-model-ddgan-failure-investigation-v1.json)

### api-model-vae-failure-investigation · api-model-vae-failure-investigation-v1

**Scope:** historical; family_context; revision `94f596832d94bce175b5d4d6653b3584e17f6dbae570113717d570deb431a9b4`.

Hard routing leaves more uniform-prior rows weakly constrained by reconstruction; those rows account for larger independent-generation errors.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `saved_state_failure_analysis`.



All archived final metrics reproduce exactly with zero updates. All 183 hard decoder-mean misses arise from 12 low-usage rows; the passing categorical control has the same association with 26 misses from seven low-usage rows. Hard ELBO prior gradients reach 4/16 rows versus 16/16 for categorical; adversarial and SBNP reach all rows. This supports a final-state coverage association but cannot establish the cause of the unsaved update-1000/1100 failures. Original hard FAIL/categorical PASS and all gates remain unchanged.

**Next:** Retain both outcomes. Propose one reconstruction-coverage mechanism across both unchanged VAE tasks with saved scheduled checkpoints before another causal experiment. No unchanged rerun, seed sweep, gate change or default promotion.

[Evidence](../../reports/forge/vae-failure-investigation/README.md) · [Record](records/api-model-vae-failure-investigation-v1.json)

### pr155-e22-api-and-original-review-gaps · archive-pr155-current-api-and-review-gaps-183242edb5fb

**Scope:** historical; family_context; revision `183242edb5fb178a11982fe0f38bc775764f9be47cc802b3a98749ca5a44b96b`.

New E22/KA2 API and archived research must retain their original particle, serving, initialization and scoring identity before comparison with Forge learned MoG.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `api_and_evidence_context`.



Incoming PR155 E22 positives cannot fill current MoG/clean-live cells: it requires ParticlePrior, noisy state-selected serving, different initialization/RNG and package. KA2 has a fixed call-800 phase change. Future integration must preserve MoG, named streams, initialization, enumeration, external budgets and complete policy checkpoints, and must avoid the empty-overrides k3p card silently adopting a KA2 default. No current-develop software defect was demonstrated. The newer archive still does not bind original dt07514k, EMA .995 D-tracking3/3 or sub-.03sigma experimental-sensitivity claims; a later written sensitivity assertion and separate EMA failure are preserved without substitution.

**Next:** Use the linked compatibility checklist when the API lands in develop. Recover original missing evidence and freeze a new justified calibration profile before default adoption. This context record launches no training and carries no current qualification.

[Evidence](../../reports/forge/UPSTREAM_E22_COMPATIBILITY.md) · [Record](records/archive-pr155-current-api-and-review-gaps-183242edb5fb.json)

### context:pr155-current:dtrack-floors · archive-pr155-current-context-dtrack-floors-97e97b2f0175

**Scope:** historical; family_context; revision `55c07d30aba1b915fbd185cf87d8e65cb6abe77e82e9b7fbb32b0d636500b392`.

Historical discriminator-rate floors and complete five-candidate comparison.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_diagnostic_context`.



Narrative16-gate matrix; no variant clears all three natives. Image failures and remote evidence gaps preserved. No raw scientific task grades inferred.

**Next:** Read complete original evidence, failed variants and provenance gaps; keep uncertain schemas as context and obtain compatible evidence before qualification.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/dtrack-floors/README.md) · [Record](records/archive-pr155-current-context-dtrack-floors-97e97b2f0175.json)

### context:pr155-current:e17-feature-gauge-review · archive-pr155-current-context-e17-feature-gauge-review-0527796ee1a3

**Scope:** historical; family_context; revision `5e9057bdad36e9f7872d67c17a2b01fddd1b0501f8148b1bf07b455dcde81aee`.

Function-preserving critic-feature gauge counterexample and limited whitening diagnostic.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_diagnostic_context`.



Hidden-axis scaling changes44 isolation moves to0 while scalar scores are unchanged. Whitening is an invariance diagnostic, not validated native improvement.

**Next:** Read complete original evidence, failed variants and provenance gaps; keep uncertain schemas as context and obtain compatible evidence before qualification.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/e17-feature-gauge-review/RESULTS.md) · [Record](records/archive-pr155-current-context-e17-feature-gauge-review-0527796ee1a3.json)

### context:pr155-current:noout-e4 · archive-pr155-current-context-noout-e4-2d33dd7e870c

**Scope:** historical; family_context; revision `21b670986297192d73a388d47050458825c8235ee39c070ac3f26f2ade50bb6f`.

E4 package and complete mixed-suite context.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_diagnostic_context`.



18 native receipts:13 noisy PASS/5FAIL. Original22-task matrix12PASS/2FAIL/8rawERROR, with all8 errors capability-parity BLOCKED. Ring/stationary supplements pass; no current qualification.

**Next:** Read complete original evidence, failed variants and provenance gaps; keep uncertain schemas as context and obtain compatible evidence before qualification.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/README.md) · [Record](records/archive-pr155-current-context-noout-e4-2d33dd7e870c.json)

### context:pr155-current:noout-support-test · archive-pr155-current-context-noout-support-test-160ddc653f36

**Scope:** historical; family_context; revision `27b72bcce96025dfd4936737db1d3abc1a1301d8aca5aab8c830f3978e5af6af`.

E14–E22 support study, including failed variants and execution errors.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_diagnostic_context`.



75 native receipts:69 noisy PASS/6FAIL; all clean FAIL. E22 owns three7k confirmations; E19 owns its13-task suite/stress runs. Ledger55PASS/10FAIL/3OOM ERROR attempts remains context, not duplicate trials.

**Next:** Read complete original evidence, failed variants and provenance gaps; keep uncertain schemas as context and obtain compatible evidence before qualification.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/RESULTS.md) · [Record](records/archive-pr155-current-context-noout-support-test-160ddc653f36.json)

### context:pr155-current:prior-ema-relaxation · archive-pr155-current-context-prior-ema-relaxation-17178718efa9

**Scope:** historical; family_context; revision `5ba3da0268f733c48893ece3d3c3e3177171f640610fc59107ab26fe3f36ab98`.

Preserve the prior-EMA relaxation host correction and original evidence as context, without fresh normalized trials.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_training_mechanism_context`.



Prior-EMA training relaxation: diagnostic fresh-callback host recorded3/3; original cached-callback canonical staggered100 FAIL (4/5 terminal accuracy, final centre.20266sigma). Distinct hosts; no frozen-host3/3, no current MoG credit.

**Next:** Keep corrected diagnostic and canonical hosts separate; consult unchanged prior audit for outstanding source gaps. No current qualification or new run authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/prior-ema-relaxation/README.md) · [Record](records/archive-pr155-current-context-prior-ema-relaxation-17178718efa9.json)

### context:pr155-current:streaming-smallbatch-toy · archive-pr155-current-context-streaming-smallbatch-toy-756ef378d97a

**Scope:** historical; family_context; revision `33b34f69ab052077167ed3c4b3186a06a7f59a191bb27936cb52038afc976ecc`.

Small-batch moving-critic memory, flow, mass and optimism diagnostics.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_diagnostic_context`.



Negative controller readout across six fixed toy cases; optimistic_go=false. Oracle arms remain evaluator-only. No native performance or qualification inferred.

**Next:** Read complete original evidence, failed variants and provenance gaps; keep uncertain schemas as context and obtain compatible evidence before qualification.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/streaming-smallbatch-toy/RESULTS.md) · [Record](records/archive-pr155-current-context-streaming-smallbatch-toy-756ef378d97a.json)

### archive:pr155:E13slr075 · archive-pr155-current-e13slr075-2706550e61f4

**Scope:** historical; scientific; revision `efb4fa10cd0a26658ef1a509bada1069c761e09617b55ca2bcc8733c5663290e`.

Historical native evaluation of declared E13slr075 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 5585.6; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98455, precision=0.98455, acc_center_rms_sigma=0.15904, mass_tv=0.03405, acc_radial_ks=0.014007; rotated100: modes=100, hq=0.97415, precision=0.97415, acc_center_rms_sigma=0.11769, mass_tv=0.0301, acc_radial_ks=0.0096887; staggered100: modes=100, hq=0.97665, precision=0.97665, acc_center_rms_sigma=0.13944, mass_tv=0.0384, acc_radial_ks=0.0067139

Historical E13slr075: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E13slr075-grid100/result.json) · [Record](records/archive-pr155-current-e13slr075-2706550e61f4.json)

### archive:pr155:E13slr133 · archive-pr155-current-e13slr133-e77bb1764d51

**Scope:** historical; scientific; revision `ac7b08747771ced916ac50ca69f326a796f4e2398197d2fa180e13f1459156d4`.

Historical native evaluation of declared E13slr133 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 5431.97; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9827, precision=0.9827, acc_center_rms_sigma=0.14808, mass_tv=0.0301, acc_radial_ks=0.0095836; rotated100: modes=100, hq=0.9743, precision=0.9743, acc_center_rms_sigma=0.13944, mass_tv=0.03325, acc_radial_ks=0.01288; staggered100: modes=100, hq=0.97845, precision=0.97845, acc_center_rms_sigma=0.14066, mass_tv=0.032, acc_radial_ks=0.0056958

Historical E13slr133: 2 noisy native PASS / 1 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E13slr133-grid100/result.json) · [Record](records/archive-pr155-current-e13slr133-e77bb1764d51.json)

### archive:pr155:E14s · archive-pr155-current-e14s-d2d30216af24

**Scope:** historical; scientific; revision `c65b73b29d1160cbf9941bef626a4f28828910ecb27b6653246499bfd89d6ac0`.

Historical native evaluation of declared E14s formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 5158.37; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9837, precision=0.9837, acc_center_rms_sigma=0.13819, mass_tv=0.0345, acc_radial_ks=0.010316; rotated100: modes=100, hq=0.97415, precision=0.97415, acc_center_rms_sigma=0.12404, mass_tv=0.03205, acc_radial_ks=0.013816; staggered100: modes=100, hq=0.9758, precision=0.9758, acc_center_rms_sigma=0.1508, mass_tv=0.0343, acc_radial_ks=0.0062391

Historical E14s: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E14s-grid100/result.json) · [Record](records/archive-pr155-current-e14s-d2d30216af24.json)

### archive:pr155:E14s14k · archive-pr155-current-e14s14k-3b22f928adb5

**Scope:** historical; scientific; revision `e1a96918dd673535ec7a2c9ba1d1f4f844a29ba49b5c5f76e1f8d6e3af01145d`.

Historical native evaluation of declared E14s14k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 9221.41; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98365, precision=0.98365, acc_center_rms_sigma=0.14721, mass_tv=0.03305, acc_radial_ks=0.0064912; rotated100: modes=100, hq=0.97965, precision=0.97965, acc_center_rms_sigma=0.11102, mass_tv=0.0311, acc_radial_ks=0.014617; staggered100: modes=100, hq=0.97785, precision=0.97785, acc_center_rms_sigma=0.14618, mass_tv=0.0329, acc_radial_ks=0.0051406

Historical E14s14k: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E14s14k-grid100/result.json) · [Record](records/archive-pr155-current-e14s14k-3b22f928adb5.json)

### archive:pr155:E14s28k · archive-pr155-current-e14s28k-920d28403b8d

**Scope:** historical; scientific; revision `ddd488c38e6f65fae5af7175a5d89ba214ddf4abbe3a8a5254d5917fc03e7c54`.

Historical native evaluation of declared E14s28k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 12681.89; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9847, precision=0.9847, acc_center_rms_sigma=0.1681, mass_tv=0.0323, acc_radial_ks=0.01042; rotated100: modes=100, hq=0.97465, precision=0.97465, acc_center_rms_sigma=0.11326, mass_tv=0.0327, acc_radial_ks=0.012957; staggered100: modes=100, hq=0.9779, precision=0.9779, acc_center_rms_sigma=0.1725, mass_tv=0.02995, acc_radial_ks=0.0068476

Historical E14s28k: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E14s28k-grid100/result.json) · [Record](records/archive-pr155-current-e14s28k-920d28403b8d.json)

### archive:pr155:E15a · archive-pr155-current-e15a-aa745d297292

**Scope:** historical; scientific; revision `39d839b6ad1c025d3ebd38cd7ca6968acb0d963c0e5191cdc0d9c9ea4df95f68`.

Historical native evaluation of declared E15a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 3}; wall seconds 2371.59; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98175, precision=0.98175, acc_center_rms_sigma=0.16245, mass_tv=0.03275, acc_radial_ks=0.011912; rotated100: modes=100, hq=0.9799, precision=0.9799, acc_center_rms_sigma=0.23678, mass_tv=0.04225, acc_radial_ks=0.031748; staggered100: modes=100, hq=0.9808, precision=0.9808, acc_center_rms_sigma=0.36495, mass_tv=0.043, acc_radial_ks=0.04529

Historical E15a: 0 noisy native PASS / 3 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E15a-grid100/result.json) · [Record](records/archive-pr155-current-e15a-aa745d297292.json)

### archive:pr155:E16a · archive-pr155-current-e16a-95619b2eac2c

**Scope:** historical; scientific; revision `81cb8abb4db21d0996d6a8de6ca22df17073537b40b84be1d9df81e5768d26f0`.

Historical native evaluation of declared E16a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 2123.39; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98475, precision=0.98475, acc_center_rms_sigma=0.17144, mass_tv=0.0326, acc_radial_ks=0.01105; rotated100: modes=100, hq=0.98145, precision=0.98145, acc_center_rms_sigma=0.13896, mass_tv=0.0377, acc_radial_ks=0.014741; staggered100: modes=100, hq=0.9837, precision=0.9837, acc_center_rms_sigma=0.16093, mass_tv=0.03515, acc_radial_ks=0.014619

Historical E16a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E16a-grid100/result.json) · [Record](records/archive-pr155-current-e16a-95619b2eac2c.json)

### archive:pr155:E16b · archive-pr155-current-e16b-aac85037d724

**Scope:** historical; scientific; revision `05b9e5c9904b58b2748f9352be7e4c5a81233700fa54df167a0ea2bda9f42e38`.

Historical native evaluation of declared E16b formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 2045.65; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98385, precision=0.98385, acc_center_rms_sigma=0.1759, mass_tv=0.0328, acc_radial_ks=0.010439; rotated100: modes=100, hq=0.981, precision=0.981, acc_center_rms_sigma=0.14276, mass_tv=0.03505, acc_radial_ks=0.015343; staggered100: modes=100, hq=0.98065, precision=0.98065, acc_center_rms_sigma=0.15918, mass_tv=0.0339, acc_radial_ks=0.016251

Historical E16b: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E16b-grid100/result.json) · [Record](records/archive-pr155-current-e16b-aac85037d724.json)

### archive:pr155:E17a · archive-pr155-current-e17a-37929e6e3af0

**Scope:** historical; scientific; revision `7ea525c478b52a32675f44110245966419859ea0f8a05779936437619c520944`.

Historical native evaluation of declared E17a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 2539.97; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9836, precision=0.9836, acc_center_rms_sigma=0.14281, mass_tv=0.03465, acc_radial_ks=0.01142; rotated100: modes=100, hq=0.987, precision=0.987, acc_center_rms_sigma=0.12662, mass_tv=0.0353, acc_radial_ks=0.009245; staggered100: modes=100, hq=0.9821, precision=0.9821, acc_center_rms_sigma=0.16002, mass_tv=0.03375, acc_radial_ks=0.014499

Historical E17a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E17a-grid100/result.json) · [Record](records/archive-pr155-current-e17a-37929e6e3af0.json)

### archive:pr155:E17b · archive-pr155-current-e17b-5d0cc5aaef72

**Scope:** historical; scientific; revision `afea89f3a87e26e28ad4f00f6da55e1da0f50e8724081b1e77e1c53cb862f088`.

Historical native evaluation of declared E17b formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 3808.81; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98405, precision=0.98405, acc_center_rms_sigma=0.14767, mass_tv=0.0346, acc_radial_ks=0.011566; rotated100: modes=100, hq=0.98365, precision=0.98365, acc_center_rms_sigma=0.15336, mass_tv=0.03635, acc_radial_ks=0.012915; staggered100: modes=100, hq=0.9814, precision=0.9814, acc_center_rms_sigma=0.15163, mass_tv=0.0352, acc_radial_ks=0.015092

Historical E17b: 2 noisy native PASS / 1 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E17b-grid100/result.json) · [Record](records/archive-pr155-current-e17b-5d0cc5aaef72.json)

### archive:pr155:E17blr075 · archive-pr155-current-e17blr075-f3630311cf48

**Scope:** historical; scientific; revision `c7c567cb0ddb5e8c72ba272e0d57f4f59fe8b0bf234303bdf2e9147b61015c03`.

Historical native evaluation of declared E17blr075 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 4132.47; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9855, precision=0.9855, acc_center_rms_sigma=0.12544, mass_tv=0.0321, acc_radial_ks=0.009832; rotated100: modes=100, hq=0.9824, precision=0.9824, acc_center_rms_sigma=0.12423, mass_tv=0.03435, acc_radial_ks=0.01018; staggered100: modes=100, hq=0.98055, precision=0.98055, acc_center_rms_sigma=0.14162, mass_tv=0.03855, acc_radial_ks=0.015679

Historical E17blr075: 2 noisy native PASS / 1 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E17blr075-grid100/result.json) · [Record](records/archive-pr155-current-e17blr075-f3630311cf48.json)

### archive:pr155:E17lr075 · archive-pr155-current-e17lr075-e2f3d53275b1

**Scope:** historical; scientific; revision `193dc25ed1f75f90201d771d3a9b4b76b3b89f35c0515d8882072caea650787a`.

Historical native evaluation of declared E17lr075 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 4597.83; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9843, precision=0.9843, acc_center_rms_sigma=0.13318, mass_tv=0.03225, acc_radial_ks=0.011114; rotated100: modes=100, hq=0.98655, precision=0.98655, acc_center_rms_sigma=0.12281, mass_tv=0.03465, acc_radial_ks=0.0098251; staggered100: modes=100, hq=0.98165, precision=0.98165, acc_center_rms_sigma=0.14641, mass_tv=0.0377, acc_radial_ks=0.011877

Historical E17lr075: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E17lr075-grid100/result.json) · [Record](records/archive-pr155-current-e17lr075-e2f3d53275b1.json)

### archive:pr155:E17lr133 · archive-pr155-current-e17lr133-6b9e5a03f19e

**Scope:** historical; scientific; revision `ec4e9345f9c0c55cb70e6f04cb1651e4da1d8f2542f3fa9e28469c417b8924ae`.

Historical native evaluation of declared E17lr133 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 4540.46; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98275, precision=0.98275, acc_center_rms_sigma=0.15407, mass_tv=0.0349, acc_radial_ks=0.010613; rotated100: modes=100, hq=0.98495, precision=0.98495, acc_center_rms_sigma=0.16763, mass_tv=0.03725, acc_radial_ks=0.0146; staggered100: modes=100, hq=0.9844, precision=0.9844, acc_center_rms_sigma=0.13906, mass_tv=0.0325, acc_radial_ks=0.010772

Historical E17lr133: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E17lr133-grid100/result.json) · [Record](records/archive-pr155-current-e17lr133-6b9e5a03f19e.json)

### archive:pr155:E18a · archive-pr155-current-e18a-f2ed22a78a3f

**Scope:** historical; scientific; revision `dfa40b4ae1bf5c3f8d161afb2eb503bc1d337bfaab2daaeb8ac3dbc4155b2cd7`.

Historical native evaluation of declared E18a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 4241.16; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98535, precision=0.98535, acc_center_rms_sigma=0.14699, mass_tv=0.0344, acc_radial_ks=0.0084099; rotated100: modes=100, hq=0.986, precision=0.986, acc_center_rms_sigma=0.1447, mass_tv=0.03465, acc_radial_ks=0.015468; staggered100: modes=100, hq=0.98265, precision=0.98265, acc_center_rms_sigma=0.15442, mass_tv=0.03555, acc_radial_ks=0.015639

Historical E18a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E18a-grid100/result.json) · [Record](records/archive-pr155-current-e18a-f2ed22a78a3f.json)

### archive:pr155:E19a · archive-pr155-current-e19a-cbad326e890d

**Scope:** historical; scientific; revision `34b58cc8768ff33729a509a4e50249ad3ad4cba127e84f5b57b5366f4c056bd5`.

Historical native evaluation of declared E19a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 4907.79; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98485, precision=0.98485, acc_center_rms_sigma=0.15501, mass_tv=0.0322, acc_radial_ks=0.010549; rotated100: modes=100, hq=0.98575, precision=0.98575, acc_center_rms_sigma=0.15118, mass_tv=0.0334, acc_radial_ks=0.0099742; staggered100: modes=100, hq=0.98265, precision=0.98265, acc_center_rms_sigma=0.15035, mass_tv=0.0325, acc_radial_ks=0.012647

Historical E19a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E19a-grid100/result.json) · [Record](records/archive-pr155-current-e19a-cbad326e890d.json)

### archive:pr155:E19a14k · archive-pr155-current-e19a14k-2ea19c1c94ed

**Scope:** historical; scientific; revision `db3032d76b86a7feb4ee8e2963ce20f5b50a023554421d68d6b0f52d897167aa`.

Historical native evaluation of declared E19a14k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 14862.66; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9857, precision=0.9857, acc_center_rms_sigma=0.12943, mass_tv=0.032, acc_radial_ks=0.0045448; rotated100: modes=100, hq=0.98725, precision=0.98725, acc_center_rms_sigma=0.11394, mass_tv=0.03305, acc_radial_ks=0.009831; staggered100: modes=100, hq=0.9868, precision=0.9868, acc_center_rms_sigma=0.1485, mass_tv=0.03125, acc_radial_ks=0.0072842

Historical E19a14k: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E19a14k-grid100/result.json) · [Record](records/archive-pr155-current-e19a14k-2ea19c1c94ed.json)

### archive:pr155:E19a28k · archive-pr155-current-e19a28k-eed6161f41d4

**Scope:** historical; scientific; revision `02b8d81405a40dcd8a2e9d69dfa0ca1ed842265be71a8ea679e88f5c94fae3a2`.

Historical native evaluation of declared E19a28k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 22344.55; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9833, precision=0.9833, acc_center_rms_sigma=0.13782, mass_tv=0.03205, acc_radial_ks=0.0059623; rotated100: modes=100, hq=0.98935, precision=0.98935, acc_center_rms_sigma=0.1112, mass_tv=0.0313, acc_radial_ks=0.0048636; staggered100: modes=100, hq=0.9866, precision=0.9866, acc_center_rms_sigma=0.12328, mass_tv=0.02995, acc_radial_ks=0.0091866

Historical E19a28k: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E19a28k-grid100/result.json) · [Record](records/archive-pr155-current-e19a28k-eed6161f41d4.json)

### archive:pr155:E19lr075 · archive-pr155-current-e19lr075-f37f95a58a03

**Scope:** historical; scientific; revision `b932c7fe2c21babb4db7f4174b2224c96c989966bf0d54171bd468a7348aec8a`.

Historical native evaluation of declared E19lr075 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 6513.29; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98325, precision=0.98325, acc_center_rms_sigma=0.14193, mass_tv=0.0358, acc_radial_ks=0.011271; rotated100: modes=100, hq=0.98485, precision=0.98485, acc_center_rms_sigma=0.12852, mass_tv=0.0329, acc_radial_ks=0.012239; staggered100: modes=100, hq=0.98445, precision=0.98445, acc_center_rms_sigma=0.13579, mass_tv=0.03835, acc_radial_ks=0.011446

Historical E19lr075: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E19lr075-grid100/result.json) · [Record](records/archive-pr155-current-e19lr075-f37f95a58a03.json)

### archive:pr155:E19lr133 · archive-pr155-current-e19lr133-68b16d78a3f1

**Scope:** historical; scientific; revision `6e93cd1f1df3e931888bae6d8d52a4c79427efe614be635834679dd8ad74f149`.

Historical native evaluation of declared E19lr133 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 8308.26; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9846, precision=0.9846, acc_center_rms_sigma=0.18377, mass_tv=0.03815, acc_radial_ks=0.015429; rotated100: modes=100, hq=0.9869, precision=0.9869, acc_center_rms_sigma=0.16564, mass_tv=0.03425, acc_radial_ks=0.0087955; staggered100: modes=100, hq=0.98165, precision=0.98165, acc_center_rms_sigma=0.15749, mass_tv=0.02995, acc_radial_ks=0.01316

Historical E19lr133: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E19lr133-grid100/result.json) · [Record](records/archive-pr155-current-e19lr133-68b16d78a3f1.json)

### archive:pr155:E20a · archive-pr155-current-e20a-bd1af8036c7f

**Scope:** historical; scientific; revision `f112349486e6be51efd37024c2b34e723a682e21bbba1d8aa365d724a7a19c97`.

Historical native evaluation of declared E20a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 6483.78; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98315, precision=0.98315, acc_center_rms_sigma=0.13963, mass_tv=0.03545, acc_radial_ks=0.012222; rotated100: modes=100, hq=0.985, precision=0.985, acc_center_rms_sigma=0.14057, mass_tv=0.03295, acc_radial_ks=0.013276; staggered100: modes=100, hq=0.98185, precision=0.98185, acc_center_rms_sigma=0.16876, mass_tv=0.0316, acc_radial_ks=0.01828

Historical E20a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E20a-grid100/result.json) · [Record](records/archive-pr155-current-e20a-bd1af8036c7f.json)

### archive:pr155:E20a14k · archive-pr155-current-e20a14k-f83a60f4242e

**Scope:** historical; scientific; revision `37662488573a60096f914381c7e3f72d5f883a20b69661ddc205830d817c41c6`.

Historical native evaluation of declared E20a14k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 14635.52; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9857, precision=0.9857, acc_center_rms_sigma=0.13157, mass_tv=0.0339, acc_radial_ks=0.012589; rotated100: modes=100, hq=0.98715, precision=0.98715, acc_center_rms_sigma=0.11638, mass_tv=0.03245, acc_radial_ks=0.0048333; staggered100: modes=100, hq=0.98455, precision=0.98455, acc_center_rms_sigma=0.17832, mass_tv=0.0301, acc_radial_ks=0.012596

Historical E20a14k: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E20a14k-grid100/result.json) · [Record](records/archive-pr155-current-e20a14k-f83a60f4242e.json)

### archive:pr155:E20lr075 · archive-pr155-current-e20lr075-22307b00517a

**Scope:** historical; scientific; revision `3bf0de7f789dd7c8f7b6c3d0aff3190af1462d8d3711f7605c0896c69d858f52`.

Historical native evaluation of declared E20lr075 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 10561.34; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9844, precision=0.9844, acc_center_rms_sigma=0.14831, mass_tv=0.0362, acc_radial_ks=0.0092442; rotated100: modes=100, hq=0.98515, precision=0.98515, acc_center_rms_sigma=0.12801, mass_tv=0.0357, acc_radial_ks=0.015743; staggered100: modes=100, hq=0.98625, precision=0.98625, acc_center_rms_sigma=0.1375, mass_tv=0.038, acc_radial_ks=0.012631

Historical E20lr075: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E20lr075-grid100/result.json) · [Record](records/archive-pr155-current-e20lr075-22307b00517a.json)

### archive:pr155:E20lr133 · archive-pr155-current-e20lr133-518afdfd38b1

**Scope:** historical; scientific; revision `5496a9b5230cce459d69c4ffce3098ddb0b3c40f34586e6bd03cac70e1c455ce`.

Historical native evaluation of declared E20lr133 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 14260.74; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98155, precision=0.98155, acc_center_rms_sigma=0.15773, mass_tv=0.03785, acc_radial_ks=0.014306; rotated100: modes=100, hq=0.98655, precision=0.98655, acc_center_rms_sigma=0.161, mass_tv=0.03405, acc_radial_ks=0.010593; staggered100: modes=100, hq=0.9838, precision=0.9838, acc_center_rms_sigma=0.1477, mass_tv=0.03055, acc_radial_ks=0.010139

Historical E20lr133: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E20lr133-grid100/result.json) · [Record](records/archive-pr155-current-e20lr133-518afdfd38b1.json)

### archive:pr155:E21a · archive-pr155-current-e21a-2c37c1a4caac

**Scope:** historical; scientific; revision `716c5d2028c51f58a087239df13bc20becb27f7501d231e1a24c79b9e87c4f19`.

Historical native evaluation of declared E21a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 4980.05; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9855, precision=0.9855, acc_center_rms_sigma=0.12528, mass_tv=0.0326, acc_radial_ks=0.010988; rotated100: modes=100, hq=0.98465, precision=0.98465, acc_center_rms_sigma=0.13707, mass_tv=0.0344, acc_radial_ks=0.012816; staggered100: modes=100, hq=0.98215, precision=0.98215, acc_center_rms_sigma=0.15342, mass_tv=0.0345, acc_radial_ks=0.016606

Historical E21a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E21a-grid100/result.json) · [Record](records/archive-pr155-current-e21a-2c37c1a4caac.json)

### archive:pr155:E22a · archive-pr155-current-e22a-49ce05f26e7a

**Scope:** historical; scientific; revision `778f5c4d357e84e5d767288e1e19aef07864f3ad252878f84ea1d70fb055c14b`.

Historical native evaluation of declared E22a formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 8708.02; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98485, precision=0.98485, acc_center_rms_sigma=0.15501, mass_tv=0.0322, acc_radial_ks=0.010549; rotated100: modes=100, hq=0.98575, precision=0.98575, acc_center_rms_sigma=0.15118, mass_tv=0.0334, acc_radial_ks=0.0099742; staggered100: modes=100, hq=0.98265, precision=0.98265, acc_center_rms_sigma=0.15035, mass_tv=0.0325, acc_radial_ks=0.012647

Historical E22a: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=served_average (4 served blocks). No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-support-test/receipts/E22a-grid100/result.json) · [Record](records/archive-pr155-current-e22a-49ce05f26e7a.json)

### archive:pr155:E4 · archive-pr155-current-e4-e70a165ba0fe

**Scope:** historical; scientific; revision `8f992daece6970f5c4cad4285a6c2dff655ec98a119156f85f0d344346fd3bd2`.

Historical native evaluation of declared E4 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 5466.53; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9832, precision=0.9832, acc_center_rms_sigma=0.18667, mass_tv=0.0304, acc_radial_ks=0.013806; rotated100: modes=100, hq=0.9732, precision=0.9732, acc_center_rms_sigma=0.15313, mass_tv=0.03, acc_radial_ks=0.0096144; staggered100: modes=100, hq=0.976, precision=0.976, acc_center_rms_sigma=0.1772, mass_tv=0.03455, acc_radial_ks=0.0075883

Historical E4: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=recorded_fast_live. No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/native-receipts/E4-grid100/result.json) · [Record](records/archive-pr155-current-e4-e70a165ba0fe.json)

### archive:pr155:E414k · archive-pr155-current-e414k-fb136a05fcde

**Scope:** historical; scientific; revision `2c1bcc4766965de510380610ee305f47c5204690ffd05b056747cc95c505b5b3`.

Historical native evaluation of declared E414k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 6516.3; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9825, precision=0.9825, acc_center_rms_sigma=0.17291, mass_tv=0.03075, acc_radial_ks=0.010435; rotated100: modes=100, hq=0.9723, precision=0.9723, acc_center_rms_sigma=0.14756, mass_tv=0.0303, acc_radial_ks=0.0051042; staggered100: modes=100, hq=0.9777, precision=0.9777, acc_center_rms_sigma=0.15402, mass_tv=0.03495, acc_radial_ks=0.0048961

Historical E414k: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=recorded_fast_live. No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/native-receipts/E414k-grid100/result.json) · [Record](records/archive-pr155-current-e414k-fb136a05fcde.json)

### archive:pr155:E4lr075 · archive-pr155-current-e4lr075-75e372d94a24

**Scope:** historical; scientific; revision `f6c28e8c56ad36dc821a2f945f86beb349fe7e21e67b40fefa3a3325b351c405`.

Historical native evaluation of declared E4lr075 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'PASS': 3}; wall seconds 3869.2; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98465, precision=0.98465, acc_center_rms_sigma=0.15091, mass_tv=0.02995, acc_radial_ks=0.0097769; rotated100: modes=100, hq=0.97555, precision=0.97555, acc_center_rms_sigma=0.15867, mass_tv=0.02785, acc_radial_ks=0.0078124; staggered100: modes=100, hq=0.97315, precision=0.97315, acc_center_rms_sigma=0.17406, mass_tv=0.03505, acc_radial_ks=0.011002

Historical E4lr075: 3 noisy native PASS / 0 FAIL; all three clean diagnostics FAIL. Scoring=recorded_fast_live. No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/native-receipts/E4lr075-grid100/result.json) · [Record](records/archive-pr155-current-e4lr075-75e372d94a24.json)

### archive:pr155:E4lr133 · archive-pr155-current-e4lr133-be7aab4f2a56

**Scope:** historical; scientific; revision `72fb9ab6b469191c06ac0bc4490a5fe7f8c52b4e223c56e2f503b8563d6e9b21`.

Historical native evaluation of declared E4lr133 formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 3}; wall seconds 3079.44; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9766, precision=0.9766, acc_center_rms_sigma=0.20929, mass_tv=0.0352, acc_radial_ks=0.025273; rotated100: modes=100, hq=0.976, precision=0.976, acc_center_rms_sigma=0.18501, mass_tv=0.02875, acc_radial_ks=0.0050108; staggered100: modes=100, hq=0.9765, precision=0.9765, acc_center_rms_sigma=0.20463, mass_tv=0.02915, acc_radial_ks=0.0092151

Historical E4lr133: 0 noisy native PASS / 3 FAIL; all three clean diagnostics FAIL. Scoring=recorded_fast_live. No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/native-receipts/E4lr133-grid100/result.json) · [Record](records/archive-pr155-current-e4lr133-be7aab4f2a56.json)

### archive:pr155:E4nh · archive-pr155-current-e4nh-9d90739f8c00

**Scope:** historical; scientific; revision `b789258511dc34f0c876975fb257849b032f1412c93092231512bbdfc5901c90`.

Historical native evaluation of declared E4nh formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 5074.82; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.98315, precision=0.98315, acc_center_rms_sigma=0.19756, mass_tv=0.02995, acc_radial_ks=0.015494; rotated100: modes=100, hq=0.97785, precision=0.97785, acc_center_rms_sigma=0.14378, mass_tv=0.0274, acc_radial_ks=0.010283; staggered100: modes=100, hq=0.976, precision=0.976, acc_center_rms_sigma=0.1772, mass_tv=0.03455, acc_radial_ks=0.0075883

Historical E4nh: 2 noisy native PASS / 1 FAIL; all three clean diagnostics FAIL. Scoring=recorded_fast_live. No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/native-receipts/E4nh-grid100/result.json) · [Record](records/archive-pr155-current-e4nh-9d90739f8c00.json)

### archive:pr155:E4nh14k · archive-pr155-current-e4nh14k-e4138ef980fe

**Scope:** historical; scientific; revision `48af2ce32bc61335ed1cb7321596bc447e891eb99a07d9bc2653a25d827d30a1`.

Historical native evaluation of declared E4nh14k formulation under its archived fixed fixture; mechanisms and limitations are in the bundled family report.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 5769.08; mechanism `historical_training_and_serving_formulation`.

grid100: modes=100, hq=0.9826, precision=0.9826, acc_center_rms_sigma=0.19558, mass_tv=0.02965, acc_radial_ks=0.016336; rotated100: modes=100, hq=0.97035, precision=0.97035, acc_center_rms_sigma=0.19779, mass_tv=0.02755, acc_radial_ks=0.013166; staggered100: modes=100, hq=0.9746, precision=0.9746, acc_center_rms_sigma=0.17049, mass_tv=0.03505, acc_radial_ks=0.010286

Historical E4nh14k: 2 noisy native PASS / 1 FAIL; all three clean diagnostics FAIL. Scoring=recorded_fast_live. No current learned-MoG qualification credit.

**Next:** Recall under exact original package, recipe, fixture and serving law; retain failures and limitations. No qualification reuse or new run is authorized.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/da3b0470918fc0045175441ce24e69777b5d5990/reports/toy100/lrfree-search/noout-e4/native-receipts/E4nh14k-grid100/result.json) · [Record](records/archive-pr155-current-e4nh14k-e4138ef980fe.json)

### pr261-routed-latent-noise-coverage · archive-pr261-routed-latent-noise-coverage-7943dda321c1

**Scope:** historical; family_context; revision `7943dda321c1011dd3c86fc598a92b3493159a83`.

Does growing the shared training noise support from 2 to 8 seeds improve unseen-noise conditional recovery when architecture, native game, initialization, updates and paired addresses are held fixed?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `training_latent_support_coverage`.



Yes, within the generated routed E22 FP32 fixture: growing shared training noise support from 2 to 8 seeds cut final held-out TEST RMSE 23.09%, 0.0533702454 to 0.0410462237, at 1,024 native updates per arm; all 11 preregistered gates passed. The eight-seed arm improved at all five fixed endpoints (896-1024), for every caption source and both held-out noise seeds, under both common frozen terminal critics; the TEST-minus-FIT MSE gap fell from 0.0005545459 to -0.0000674414, and zeroing its particle code raised RMSE to 0.0895439938. Both arms proposed 10 structural events and accepted zero, so this is a data-coverage effect, not a structural-controller mechanism. Limits: generated FP32 fixture with smaller dimensions than Supra; no Supra transfer tested; full-task LoRA RMSE 0.0633609846 still beats best particle 0.0679143351; no default adoption or Forge promotion. The variable is support size (228 vs 912 unique fit contexts), not a seed repeat.

**Next:** If the routed or Supra latent-coverage question is picked up again, test expanded noise support on the actual frozen Supra teacher as one declared delta. Do not repeat the generated cohort and do not change defaults.

[Evidence](../../reports/forge/continuous-baseline-20261003/OPEN_TOY_PR_REVIEW.md) · [Record](records/archive-pr261-routed-latent-noise-coverage-7943dda321c1.json)

### archive:pr320:network-spectral-cap · archive-pr320-network-spectral-cap-c4627fb14619

**Scope:** historical; family_context; revision `c4627fb14619ca0bcd1b8d904e7796d0834a4e8a`.

Preserving small G/D gradient singular values at fixed nominal rates repairs Gaussian acquisition, retention and target-shift response while the prior retains row-normalized learning.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `network_spectral_cap`.



Fixed G/D spectral cap scale .1 improves Gaussian shape but fails continuous learning in all three timing arms. Past extrapolation passes 43/96 stationary checks (41/72 retention), first five-pass window at update 3875, and 17/48 shifted checks (8/24 retention), ending shifted KS .03837; all acquisition and reacquisition suffixes are zero. Every ring arm passes 0/240 checks, versus original alternating control acquisition PASS and 144/144 retention. Stop this exact global recipe; prior sampled-row normalization was unchanged, so this does not evaluate a combined prior/network repair. Nine CUDA trials use 30000 new updates, 362.226751 loop seconds and zero scientific retries. This is a task-only diagnostic, not ordinary qualification.

**Next:** Retain the archived findings and original source; stop these exact revisions. Any successor needs its own substantive hypothesis, frozen protocol and finite budget. No training or default adoption follows from this archive.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/c4627fb14619ca0bcd1b8d904e7796d0834a4e8a/reports/forge/gaussian-network-magnitude/README.md) · [Record](records/archive-pr320-network-spectral-cap-c4627fb14619.json)

### archive:pr360:hydraulic-winner · archive-pr360-hydraulic-winner-bbac181e1281

**Scope:** historical; family_context; revision `bbac181e128160a1e5770b60490ba99ebaffbf01`.

Bound joint G/prior output travel on the direction-blend BCAP winner; compare real-spacing radius with a gap-adaptive radius, preserving every original task gate and the five winner passes in the 17-cell diagnostic scope.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `hydraulic_winner`.



Latest 17-public-trainer-task winner follow-up: travel and gap-adaptive each pass 4/17 versus archived winner 5/17, gain zero gates, and respectively lose ring16_acquisition and vector_spiral. Gaussian stationary/shift retention improves 2/72 and 0/24 to 71/72 and 23/24 (travel), or 68/72 and 22/24 (gap); full retention FAIL remains. Grid100 precision rises .24072 -> .49586/.6343 but native full shape/density gates fail. Travel exactly reproduces archived v1 on the bound hosts; direction blend is inactive there, not a fresh matched control. Both prespecified endpoint predictions are observed, but the global decision stops both revisions. 34 complete cells, zero retries, 5159.621962 paid seconds within 64000 ceiling. Earlier rounds retain separate source identities; no ordinary qualification or default change.

**Next:** Retain the archived findings and original source; stop these exact revisions. Any successor needs its own substantive hypothesis, frozen protocol and finite budget. No training or default adoption follows from this archive.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/bbac181e128160a1e5770b60490ba99ebaffbf01/reports/forge/bcap-physics/hydraulic/winner/README.md) · [Record](records/archive-pr360-hydraulic-winner-bbac181e1281.json)

### archive:pr395:four-bcap-repairs · archive-pr395-four-bcap-repairs-51ee43476e8b

**Scope:** historical; family_context; revision `51ee43476e8bfa64f0eb55f630fe3b0086dddd9b`.

Four fixed global BCAP recipe changes must retain all Tier1 gates and every matched control Tier2 pass while adding at least one complete Tier2 repair.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `four_bcap_repairs`.



No global winner. Fresh control passes 6/6 required Tier1 and 9/21 Tier2. Late annealing is 5/6, half G/E motion 4/6, prior smoothing .01 5/6; all 21 Tier2 cells for each are BLOCKED/unmeasured, not measured failures. Weak critic damping .05 passes 6/6 and 8/21, adding zero passes and losing the original word-hold gate. That legacy hold uses unequal acquired prefixes (control 834->4834, hybrid 1667->5667), so it is a separate original-gate cohort, not a same-clock causal comparison; other 20 Tier2 tasks match complete budgets/consumed streams and retain the same eight passes. Gaussian hybrid stationary/shift checks improve 2->15/72 and 0->3/24, but full gates fail. 77 paid attempts, 6773.287376 seconds, zero execution retries. Stop these four revisions; current selected recipe, qualifications and gates remain unchanged.

**Next:** Retain the archived findings and original source; stop these exact revisions. Any successor needs its own substantive hypothesis, frozen protocol and finite budget. No training or default adoption follows from this archive.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/51ee43476e8bfa64f0eb55f630fe3b0086dddd9b/reports/forge/bcap-tier2-round6/README.md) · [Record](records/archive-pr395-four-bcap-repairs-51ee43476e8b.json)

### bcap-stall-reach-base-v1 · archive-pr397-stall-reach-critic-read-de93dbf48cab

**Scope:** historical; family_context; revision `de93dbf48cabe6bb2913b3eccf93029655da5db9`.

Closed PR #107 stall reach rebuilt on the BCAP direction-blend leader (opt-in generator_critic_read="stall_reach"): the generator reads the critic through a 2d+1 axis stencil of width max(.15,.5 min(u,1/u))/kappa, .5/kappa after 50 consecutive u>=.6 critic updates, converting mode_hold FAIL to a sustained PASS while keeping all six Tier 1 gates.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `generator_critic_read_stencil`.



Falsified at Tier 1. Against its fresh matched leader control (same source cbba593bb98d, seed 0, deterministic_orthogonal, revision 9), stall_reach passes 4/6 Tier 1 versus 5/6: ring16_acquisition is lost (terminal suffix 0/5, 14/16 modes, HQ .530 versus control suffix 26, 16 modes, HQ .961, first pass 1184). Under BCAP the critic's input slope sits near its cap, so the saturation-only stall rule (PR #107's G trust factor has no current counterpart) latches: 1545/1600 ring16 updates are stalled and the wide read is on for 1597/1600, pinning the width at .5. Both arms' five_word_joint_smoke are INCOMPLETE at the 900 s allowance (control 17501/20001 updates under host CPU contention; stall_reach 10834, about 1.6x per-update cost from the 341-point read). All 21 Tier 2 tasks are BLOCKED for both arms; mode_hold is unmeasured. Public-API ring8 toy (after the review fix making its particle table deterministic): off path PASS (7 modes, HQ .917), stall_reach FAIL (6 modes, HQ .747, 1093/1200 stalled); both arms oscillate, so the toy only supports the ring16 result. PR #397 was closed unmerged; the option, its tests and the toy code are not on develop.

**Next:** Do not re-run or tune this rule (widths, windows). A successor needs a stall signal that is selective under BCAP's saturated slope and a constant-cost read (a few random or antithetic directions, not every axis); register it as a new idea with a fresh control and screen it on ring8/ring16 toys first.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/reports/forge/stall-reach/README.md) · [Record](records/archive-pr397-stall-reach-critic-read-de93dbf48cab.json)

### bcap-generator-idle-base-v1 · archive-pr398-generator-idle-se-835e93900c21

**Scope:** historical; family_context; revision `835e93900c2103eef24ed4b62e18fcdbf19e41bf`.

Closed PR #366 generator idle rule rebuilt on the recipe factories (opt-in generator_idle_se=1.0): skipping the G/prior step while the critic's paired real-fake gap is within one standard error of zero stops the constant-rate BCAP leader drifting during holds, repairing gaussian1d_stability and mode_hold without losing any control pass.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `generator_step_idle_gate`.



Falsified at Tier 1. Against its matched BCAP leader control (source digest 2ffc95c341a7, seed 0, deterministic_orthogonal, revision 9), generator_idle_se=1.0 loses ring16_acquisition PASS->FAIL (terminal suffix 0/5, final component_covariance_error 3.09991 vs limit .85; control .496567, suffix 26) after skipping only 5/1600 G steps, so all 21 Tier 2 cells are BLOCKED. The control's five_word_joint_smoke was INCOMPLETE (19168/20001 updates at the 900 s allowance under shared-GPU contention; no retry). Idle fraction: gaussian1d_smoke 42.9%, ae_gan_hold 25.6%, five_word_joint_smoke 54.6%, two_pole 1.2%, unused_token_hold 0%, ring16 0.3%. Off-ladder diagnostic (separate cohort, no credit): gaussian1d_stability improves (stationary 2->14/72, shifted hold 0->6/24, std_ratio .662->.865, cdf_ks .321->.127) but mean_error_sigma worsens .255->.312 and the gate stays FAIL; mode_hold is unchanged FAIL because the rule never idles (0/1200). PR #398 was closed unmerged; the option and its tests are not on develop.

**Next:** Do not sweep the threshold or re-run this rule. A successor must change the idle signal (it never fires on mode_hold and fires about half the time on easy holds), registered as a new structural idea with a fresh matched control.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/835e93900c2103eef24ed4b62e18fcdbf19e41bf/reports/forge/bcap-generator-idle/README.md) · [Record](records/archive-pr398-generator-idle-se-835e93900c21.json)

### bcap-default-baseline-direction-v1 · bcap-default-baseline-concluded-context-v1

**Scope:** historical; family_context; revision `unknown`.

A single global direction-blend delta preserves all six ordinary Tier 1 gates and strictly extends the matched incumbent Tier 2 pass set under the seed-0 public initializer and original task budgets, priors and sampling laws.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `protected_actual_parameter_direction`.



Completed ordinary matched comparison: direction_blend passes 6/6 Tier 1 and 9/21 Tier 2 versus none at 6/6 and 7/21, with no lost passes. Trajectory identity MSE improves from .239861891 to .000258480; residual improves from .061036013 to .000249089, with sustained suffixes 19/21 repairing their gates. Own word hold remains PASS from matched 834-step prefixes through local step 4834. All 27 required final consumed-state pairs verify; all 21 zero-activation task pairs have byte-identical actual trained tensors for every applicable role. Six tasks activate 337 blends with zero stalls. All 12 remaining Tier 2 failures are inactive and have unchanged final metrics; direction protection does not establish missing mass/density forces. Both requested Tier 3 tasks remain BLOCKED with no spend. The 56 paid attempts charge 4187.683586815016 seconds with zero retries; 56 actual-training GIFs come from saved events. The owner selected the whole measured BCAP research pin and named preset; calibration remains provisional and default_adoption=false. The full CPU run with 5938 passes and 24 failures is preserved; narrowly corrected compatibility metadata checks resolve its exact 24 failures without changing comparator bytes or numerical assertions. Reporting reconstruction restores exact admitted study authority and Git-pinned motivation, leaving numerical task statuses and original receipts unchanged. This context supplies no qualification or cross-cohort pooling.

**Next:** Use the exact direction-blend research baseline. Investigate one globally configured mass/density mechanism with an explicit numerical falsifier and measurements of actual force/displacement; preserve all six Tier 1 gates, trajectory, residual and own word hold. Inspect full/core component density, allocation and spill on the 12 remaining failures. Retain negative moonshot evidence and unavailable fields; do not rerun unchanged experiments for a merge or perform seed-only experiments.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0f5aa322a17f74ddcc08411d98fe71c9674230f5/reports/forge/bcap-default-baseline/README.md) · [Record](records/bcap-default-baseline-concluded-context-v1.json)

### bcap-moment-tradeoff-saved-diagnostic-v1 · bcap-moment-tradeoff-saved-diagnostic-v1

**Scope:** task_only_diagnostic; family_context; revision `98625b0dfc8b82176cc3b05e7c44a3637aa26664d9c4f55db91d0e745eb0ca2f`.

What distinguishes the moment-tether mode_hold repair from its five-word-hold regression?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `saved_prior_force_and_word_path_diagnosis`.



Mode repair improves the sustained quality suffix 3 to 7 against >=5. Word failure at5167 loses melon generation while paired reconstruction remains confident; at6167 generation passes while apple fourth-character reconstruction probability .796768 misses >=.9. All actual direction-blend counters remain zero. Fixed-panel tether/adversarial prior force ratio is .010873 at mode endpoint, .074770 at own word acquisition and .341457 at word endpoint. At word endpoint the direct joint-latent gradient norm is about403 times the decoder path. No historical failure-window checkpoint or gradient exists, so these endpoint diagnostics do not identify one causal explanation.

**Next:** Investigate one global relative-force limit on the covariance tether, with actual gradients/applied displacements at discriminating windows and original word hold plus all incumbent passes as controls. Hypothesis unexecuted; no default adoption or automatic continuation.

[Evidence](../../reports/forge/bcap-moment-tradeoff/results.json) · [Record](records/bcap-moment-tradeoff-saved-diagnostic-v1.json)

### bcap-three-phase-concluded-research · bcap-three-phase-concluded-context-v1

**Scope:** research_diagnostic; family_context; revision `unknown`.

Repair BCAP globally while preserving all six Tier 1 gates and the original task budgets, priors, deterministic initializer, sampling laws, gates and own-checkpoint continuation.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `source_bound_global_trainer_research`.



Direction-only projection (PR383, constraint_geometry_mode=direction_blend) is the measured research baseline: 6/6 Tier 1 and 3/10 selected Tier 2 versus control 6/6 and 1/10. Trajectory and residual student fully repair with identity MSE .000258480/.000249089 and terminal suffixes 19/21; own word hold remains PASS. All 16 complete consumed-state pairs match and 11 inactive model/prior tensor comparisons pass. Only 12/400 trajectory and 55/400 residual steps blend, with no stalls; the stored derivative maximum is floored at zero and does not prove finite-step descent. Seven studied Tier 2 gates remain FAIL, eleven original Tier 2 questions are unmeasured. Five moonshots have no eligible replacement: optimism 3/6 + 0/10, anisotropic geometry 4/6 + 1/10, entropic transport 4/6 + 2/10, bounded secant 4/6 + 0/10, confidence transport 4/6 + 2/10. Entropic/confidence each repair anisotropic density but regress pole and unused token. Aggregate moments, mass allocation and core shape can improve while full component density fails. Secant ring global covariance .098413 coexists with full component error 57.418460. Secant word is INCOMPLETE at its original 900-second limit; own hold BLOCKED. Anisotropic/entropic valid word holds use unequal own prefixes, so consumed-state pairing is unverified. Confidence alpha scales losses before the normalized optimizer, not displacement or calibrated confidence; it retains a software allowance violation (at least 2279.158 seconds against 300). The entropic constant-panel normalization amplifies squared error by 4194304; zero residual alone establishes no useful force. Five pairs charge 44305.65891298147 seconds including 18 user-authorized infrastructure predecessors; projection separately charges 3541.1636059249986 seconds with zero retries. Costs are not pooled speed comparisons. All six independent audits pass with 188 actual-training GIFs. These source-bound diagnostics grant no ordinary qualification, default adoption or automatic merge.

**Next:** Base future BCAP research on the exact PR383 direction-only recipe; stop all five exact moonshot revisions and retain their negative evidence. Inspect saved equal-clock full/core density, mass allocation, spill and controller evidence before composition; keep unavailable gradient-force measurements explicit. Any next global trainer comparison should isolate one causal mechanism with a fixed backend and numerical falsifier, measure force scaling and actual displacement, and preserve all six Tier 1 gates, trajectory, residual and word hold. A common-clock word hold would be a separately declared new cohort; never regrade these original own-prefix results.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/b235112dee4dd6d06a4e3eca7c1e31918a6a8b51/reports/forge/bcap-three-phase/README.md) · [Record](records/bcap-three-phase-concluded-context-v1.json)

### context:local-mog-envelope · context-local-mog-envelope-7f6e13219e35

**Scope:** historical; family_context; revision `7f6e13219e35948be5426b67d2248b9965941a6827ac314b13e119df54f70ade`.

Historical learned-MoG component count, relative-width, optimizer and duration study under the C0 observed envelope.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `historical_learned_mog_component_scale`.



Complete historical component-scale CSV: 26 rows, 6 C0-envelope passes and 20 failures; all 26 strict flags false. Positive-noise MoG: 4 envelope passes and 13 failures. Four exact selected receipt sets score EMA G/prior, 200k samples, seed 1, relative widths; none establishes current native or full-reference qualification. Full CSV includes nine zero-sigma cloud/atom controls (two envelope passes), while literal prior_kind is mog for every row.

**Next:** Use as historical prior art with the archived evaluator and failures visible; obtain compatible full-reference evidence before treating any lineage as a current positive. No new run is authorized by this context import.

[Evidence](../../reports/forge/supplemental/local-mog-envelope-v1/manifest.json) · [Record](records/context-local-mog-envelope-7f6e13219e35.json)

### develop-original-atlas19-and-gate-diagnosis · develop-original-atlas19-and-gate-diagnosis-fae8ce2a6d7f

**Scope:** historical; family_context; revision `fae8ce2a6d7f69f4a12d2b21b62a8444e2ad8a91ffdb1606f1fb53691772ad78`.

Determine whether post-merge gate failures changed the E22/Atlas formulation or instead tested a different candidate, prior, initialization, sampler or runtime.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `original_cohort_replay_and_sampling_contract_audit`.



Fresh, independently audited original Atlas replay passes 19/19 CUDA gates (3 native, 13 portability, 3 moving) for 48,800 updates and 6,546.906 wall seconds under current contention. All three static native endpoints exactly match the original qualified metrics. Public formulation and Atlas configuration are unchanged. PR #209 had changed native scoring to clean while transfer scoring stayed noisy; explicit sampling-law binding repairs that mixed-cohort checker. All three Atlas clean diagnostics still fail. The first corrected legacy runtime passed 22/22 locally, while the next hosted runtime truly failed 19/22 (grid100, unequal-mass and unequal-width vectors). The completed portable CPU profile also truly fails 19/22: native 3/3, transfer 16/19, with trajectory, mode hold and unequal-width vectors failing unchanged live gates. Timed local/45-minute attempts remain incomplete. Two completed portable native tasks have exact saved-draw equality across Ryzen/EPYC (82 files, 246 arrays); this supplies no aggregate or vendor-causation claim. Software CI passes 2,249 tests and 18 subtests; history/memory checks pass 37 tests. Earlier clean-MoG K3P/BCap/R1-R2/release failures and infeasible profiles remain valid. Neither the replay nor sampling repair qualifies current Forge MoG/clean-live, clock-free behavior or public-default adoption.

**Next:** Keep the integration on develop and the release branch held. Preserve every failed cohort; do not fill or tune the concluded failed profiles. Before main/default adoption, freeze a justified policy-aware qualification matching the intended prior, initialization, sampling law, serving weights and controls. No additional training is launched by this context record.

[Evidence](../../reports/develop-gates-20261001/README.md) · [Record](records/develop-original-atlas19-and-gate-diagnosis-fae8ce2a6d7f.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · five-word-tier-split-v1-readout

**Scope:** task_only_diagnostic; family_context; revision `d0ebccd0900ed102d1fa7ce028e862f53bf5e89b0e30d719950cfd6f43a4b990`.

Separate acquisition of the full joint five-word goal from retention under continuing learning, retaining all numerical thresholds and original evidence.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `task_policy_and_own_checkpoint_continuation`.



Selected BCAP DualNorm full CUDA smoke PASS:13/24 independently confirmed checks, earliest update1,667. Exact own-state hold FAIL:20/25 checks; failures at3,001/5,001/5,167/5,334/5,501; both endpoints PASS. At3,001 generation passes while paired inverse fails. Exactly24,001 research updates;930.430 loop seconds,947.754 total wall seconds,1,200 reserved.100 saved metric sets and complete checkpoints verified;372 software checks pass, including9 CUDA numerical/provenance checks. New ordinary contracts remain UNKNOWN; scientific rows and historical task bytes unchanged. One automatic cuSOLVER fallback warning; no driver change or causal claim.

**Next:** Adopt acquisition/hold placement, retain the holding failure, and use the separate saved-state diagnosis before declaring another bounded mechanism experiment. No automatic research repeat or ordinary qualification credit.

[Evidence](../../reports/forge/five-word-tier-split/readout.json) · [Record](records/five-word-tier-split-v1-readout.json)

### fivewords-saved-endpoint-dynamics-v1 · fivewords-saved-endpoint-dynamics-v1-readout

**Scope:** task_only_diagnostic; family_context; revision `2ecb68de48e8b404d472d72a860d6b4f403102f716110b2561f6778bbe652500`.

Which mechanisms explain five-word inverse/coverage instability: adversarial circulation, normalized-step pace, or moving code correspondence?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `saved_endpoint_role_and_raw_game_field_diagnosis`.



All four passing endpoints lose the exact output goal within eight G/E/prior updates with D fixed. E-only breaks the smoothed truncated/threaded inverse after one step: minimum token probability .999492 → .038852, while the joint surrogate improves .694092 → .693229. All eight first E-only steps lower the surrogate; the latent line shows local surrogate/output-goal tension and sharp decoder geometry. Two passing unsmoothed threaded raw projections are rotation-dominant (skew/symmetric 8.19 and 10.48); failed endpoints in this projection are dominated by symmetric curvature. Two zero-G-direction cases remain null. Continuing normalized E/prior motion despite saturated G changes alignment; task PASS does not certify joint equilibrium. Local sufficient causes and coexistence identified, not one historical/global cause; no qualification/default/Tier 2 credit.

**Next:** Prioritize preserving paired inverse while joint code alignment improves. A direct paired reconstruction constraint/anchor is a justified unexecuted structural option; magnitude-sensitive E/prior pace and extragradient remain hypotheses, no new study launched. Preserve global constant rates and declared controls.

[Evidence](../../reports/forge/fivewords-dynamics/readout.json) · [Record](records/fivewords-saved-endpoint-dynamics-v1-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · gaussian-no-fourier-v1-external-readout

**Scope:** task_architecture_diagnostic; scientific; revision `d0ebccd0900ed102d1fa7ce028e862f53bf5e89b0e30d719950cfd6f43a4b990`.

Removing the scalar critic's Fourier features could improve Gaussian acquisition and stability while preserving the winning BCAP learner.

**Observed:** {'FAIL': 2}; wall seconds 49.887; mechanism `task_owned_architecture`.



The bounded Fourier ablation fails the confirmed Gaussian smoke: 0 of 24 primary checks pass within 1,000 updates. Its diagnostic continuation also fails: stationary hold 3 of 72, missed shifted reacquisition deadline and shifted hold 1 of 24; frozen control 0 of 48. The final 6,000-update endpoint passes individually (KS 0.0458484), which does not replace the missed acquisition or stability gates. Exactly 6,000 CUDA updates cost 49.887372 adapter-loop seconds. Six exact restores, 219 saved-sample recomputations and two actual-training GIFs validate the published evidence. PR 323 remains the separate negative ablation; its implementation is not adopted.

**Next:** Retain Fourier 2 in the ordinary Gaussian architecture. Stop this exact Fourier 0 arm; use the independent shallow diagnostic and separate Tier 1 smoke/Tier 2 stability questions without transferring their gates across architectures. No new training, tuning, seeds or default adoption follows.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/37eb6bef4a71fd21660e4c37a781392371f59c5f/reports/forge/gaussian-no-fourier/results.json) · [Record](records/gaussian-no-fourier-v1-external-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · gaussian-shallow-v1-readout

**Scope:** task_architecture_diagnostic; scientific; revision `d079f5df4e8792c33459b08452b31b0a1136cadff2e6f3e59eddf7df890a6200`.

Reducing G/D hidden depth 2 to 1 can make a scalar Gaussian easier to acquire without changing its BCAP trainer, Fourier features, learned MoG prior, data or sampling law.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 45.755; mechanism `task_architecture_only`.



Shallow Gaussian passes the confirmed Tier 1 smoke at updates 584 and 792, but original five-terminal acquisition FAIL and Tier 2 FAIL: 4/72 stationary checks, 0/48 shift checks, 0/24 shifted hold, no deadline reacquisition. All six exact CUDA restores, 219 saved-sample metric recomputations and data/train/primary-eval stream matches pass. A self-receipt manifest bug interrupted after 1000; byte-exact own-state migration plus a source-bound software amendment completed only the remaining 5000. Exactly 6000 new updates, zero repeated prefix or scientific retries.

**Next:** Retain depth 1 as a tested diagnostic smoke variant; keep the original depth 2 ordinary Gaussian architecture because it already passes the separate confirmed smoke baseline. Do not claim convergence or adopt this as a stability repair. Stop this bounded depth arm; root merges positive diagnostic evidence and reruns the whole inventory plus eligible Tier 2 under the new view.

[Evidence](../../reports/forge/gaussian-shallow/results.json) · [Record](records/gaussian-shallow-v1-readout.json)

### gaussian-smoke-inventory-v3 · gaussian-smoke-inventory-v3-readout

**Scope:** source_bound_campaign_readout; scientific; revision `b6785b969529198b81035767e4511390f4852a7bd76bb650e755d631e03ea666`.

A confirmed can-it-pass Gaussian Tier 1 gate can unlock ordinary Tier 2 while preserving the separate continuous-stability question.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `task_gate_reform_and_source_repair`.



The exact 52-candidate CUDA roster admits 23 whole candidates. 0 pass all six required Tier 1 tasks and unlock Tier 2. All newly eligible Tier 2 jobs ran: True. 62 certified attempts cost 2897.205946 seconds; no scientific retries or cross-source task pooling. This source was interrupted; it retains missing checkpoint limitations and does not establish campaign completion. The linked readout preserves every candidate, refusal and measured scalar metric.

**Next:** Use the single regenerated technique inventory for whole-family selection; retain blocked and failed cells. No calibration, default promotion or new tuning follows from this readout.

[Evidence](../../reports/forge/gaussian-smoke-inventory/interrupted-v3/readout.json) · [Record](records/gaussian-smoke-inventory-v3-readout.json)

### gaussian-smoke-inventory-v4 · gaussian-smoke-inventory-v4-readout

**Scope:** source_bound_campaign_readout; scientific; revision `4f4e94fb1a1a8ae63afb9e930e16ae75fa53329fca85dc84d3cada4e942d4273`.

A confirmed can-it-pass Gaussian Tier 1 gate can unlock ordinary Tier 2 while preserving the separate continuous-stability question.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `task_gate_reform_and_source_repair`.



The exact 52-candidate CUDA roster admits 23 whole candidates. 0 pass all six required Tier 1 tasks and unlock Tier 2. All newly eligible Tier 2 jobs ran: True. 161 certified attempts cost 6136.617596 seconds; no scientific retries or cross-source task pooling. The linked readout preserves every candidate, refusal and measured scalar metric.

**Next:** Use the single regenerated technique inventory for whole-family selection; retain blocked and failed cells. No calibration, default promotion or new tuning follows from this readout.

[Evidence](../../reports/forge/gaussian-smoke-inventory/final-v4/readout.json) · [Record](records/gaussian-smoke-inventory-v4-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · gaussian-smoke-tier-split-v1-readout

**Scope:** task_only_diagnostic; family_context; revision `d079f5df4e8792c33459b08452b31b0a1136cadff2e6f3e59eddf7df890a6200`.

Separate can-it-pass smoke from ongoing stationary/shifted stability and independently confirm a fixed historical passing state.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `task_policy_and_saved_checkpoint_confirmation`.



Historical fixed1000 Gaussian independently confirms full bounds: KS.047590, meanerror.009982sigma, stdratio.932642; primaryKS.042974. Original five-terminal acquisition remainsFAIL. Newordinary24pairedsmoke awaits inventory; noqualification regrading. 22newGaussian contracts and93focused regression checks pass. Serialization error recovered solely fromsavedtensor/state, no repeat.

**Next:** Integrate successful smoke/stability split and architecture experiments, rerun exact ordinary current roster onCUDA, run newly eligibleTier2, regenerate the single technique-inventory. Preserve all original evidence and failedstability outcomes.

[Evidence](../../reports/forge/gaussian-smoke-tier-split/results.json) · [Record](records/gaussian-smoke-tier-split-v1-readout.json)

### k3p · gaussian1d-api-8e7f8be3b998d3daf08a0b50

**Scope:** standalone_api; scientific; revision `0a8c525384fa50273be6c67c83fd65dc0f2379a57404847917e8f7246b100d21`.

Can the public GANTrainer learn N(2, 0.5^2) from random initialization in 1,000 updates? One standalone K3P/learned-MoG cohort with 24 checks and five required terminal passes; no ordinary Forge qualification.

**Observed:** {'FAIL': 1}; wall seconds 4.953; mechanism `existing_recipe_new_scalar_host`.



Close scalar fit after 4.953 CPU seconds: mean 2.02749, std .52790 for target mean2/std.5. Strict FAIL: final KS .05674 exceeds .05, and only 3/5 terminal checks pass (12/24 overall). No recipe attribution, threshold adjustment, repeat or continuation. Oracle and seven destructive controls behave as declared; nine-state actual histogram GIF is published.

**Next:** Keep the provisional fast Tier 1 test and its FAIL. Before a more complex target, preregister one substantive stability hypothesis with explicit recipe/budget/source binding; no seed-only repeat, ordinary qualification or default adoption.

[Evidence](../../reports/toy_audit/api_contract/gaussian1d/results.json) · [Record](records/gaussian1d-api-8e7f8be3b998d3daf08a0b50.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · gaussian1d-saved-state-diagnosis-v1-readout

**Scope:** task_only_diagnostic; family_context; revision `d079f5df4e8792c33459b08452b31b0a1136cadff2e6f3e59eddf7df890a6200`.

Does the scalar failure reflect insufficient architecture/prior capacity, or loss of an already acquired fit?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `saved_state_and_fixed_representation_diagnostic`.



Actual scalar host is z_dim2, width32/depth2 (G1185/D1281 parameters), correcting the 4/64 handoff typo. All 97 saved sample metrics reproduce and all three CUDA contexts restore exactly. An affine map in the unchanged MLP with unchanged initial 256-location sigma-.1 MoG passes 24/24 numerical checks; conservative exact mixture KS bound .005086. Trained fresh 32768-sample KS is .04160 at1000 and .09307 at4000; fitted-normal shape KS worsens .03613 to .16831, skew1.602. At4000 only .47% of output variance is within components; prior displacement from initialization averages1.403. Constant normalized .012/.018/.03 steps and moving inputs support a stability hypothesis without assigning causality. Zero training updates, no task/gate revision or qualification credit.

**Next:** Require continuous acquisition, quality retention and renewed response to a target shift, without elapsed-step annealing. First propose a separately declared frozen-prior-from-initialization diagnostic with G/D learning throughout. Then compare magnitude/evidence-sensitive prior response or an explicit position spring. Existing prior_reg0 cannot be decreased; VICReg only floors spread and decorrelates, while unit row normalization can preserve motion even with a spring. A2/DV12 cannot simply be enabled for current BCAP dualnorm. Register a finite supported study and preserve matched conditions and all original gates; no new training is launched by this recommendation.

[Evidence](../../reports/forge/gaussian1d-diagnosis/results.json) · [Record](records/gaussian1d-saved-state-diagnosis-v1-readout.json)

### cf1-bdpair · history-cf1-bdpair-21ac56b45896

**Scope:** historical; scientific; revision `4b0f5c9d43e21e362f3ed0ee7a465f315ed3dbcebd1c01b4aaa9dbc25da897ef`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 724.53; mechanism `unknown`.

grid100: modes=100, hq=0.985, precision=0.985, acc_center_rms_sigma=0.19644, mass_tv=0.03235, acc_radial_ks=0.018435

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigdata-bdpair/grid100-result.json) · [Record](records/history-cf1-bdpair-21ac56b45896.json)

### cf1-bdpair · history-cf1-bdpair-b282b5feff5c

**Scope:** historical; scientific; revision `b466bbee9f26825c49293eac7b4dd058f64897ba68a6463f8fa0f94bf5277144`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 348.44; mechanism `unknown`.

staggered100: modes=100, hq=0.9761, precision=0.9761, acc_center_rms_sigma=0.17644, mass_tv=0.03225, acc_radial_ks=0.013696

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigdata-bdpair/staggered100-result.json) · [Record](records/history-cf1-bdpair-b282b5feff5c.json)

### cf1-bdpair · history-cf1-bdpair-e15258718c7c

**Scope:** historical; scientific; revision `2a6bf57878aa8fccd87443f000dce2384ca8edd70e1c7bf31936f06a90427f2a`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 348.55; mechanism `unknown`.

rotated100: modes=100, hq=0.9516, precision=0.9516, acc_center_rms_sigma=0.26623, mass_tv=0.03395, acc_radial_ks=0.063294

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigdata-bdpair/rotated100-result.json) · [Record](records/history-cf1-bdpair-e15258718c7c.json)

### context:lrfree/comparison-pr215-pr217 · history-context-lrfree-comparison-pr215-pr217-fa34f5331da4

**Scope:** historical; family_context; revision `6c128566cda4db67b4a591821d2cc61aaa33fe194efaabfda88ca39ede1c96ff`.

PR #215 GS2 and PR #217 DV12 on the #155 frozen hosts

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/README.md) · [Record](records/history-context-lrfree-comparison-pr215-pr217-fa34f5331da4.json)

### context:lrfree/critic-floor · history-context-lrfree-critic-floor-d809b4bad886

**Scope:** historical; family_context; revision `3e9324aa7bf33eaf1a45469aa286bd39a24bfaf0aca95927e60dbf57da15ee22`.

State-driven critic floor: frozen native grid100

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `training_formulation`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/critic-floor/README.md) · [Record](records/history-context-lrfree-critic-floor-d809b4bad886.json)

### context:lrfree/custom22 · history-context-lrfree-custom22-133231ccdb2b

**Scope:** historical; family_context; revision `f5f1c5384758dab8ab388fc86f5dcf718a49294e90782dbfd4cb9dea7ff76eda`.

22-check leaderboard: the eight custom hosts ported to the LR-free harness

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/README.md) · [Record](records/history-context-lrfree-custom22-133231ccdb2b.json)

### context:lrfree/direct-particle-adam · history-context-lrfree-direct-particle-adam-6a84b5d12821

**Scope:** historical; family_context; revision `cf0e41d77393b88f97bd962f0671528a964ea12259329167378a346029bdf1eb`.

Direct-particle Adam: a one-host improvement for `st-10`

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `training_formulation`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/README.md) · [Record](records/history-context-lrfree-direct-particle-adam-6a84b5d12821.json)

### context:lrfree/dtrack2 · history-context-lrfree-dtrack2-3422a7550929

**Scope:** historical; family_context; revision `3902021730de8cc93292dd8a6834868d06bb20beb3fd69d74e457920c144811d`.

Round 5: critic tracking (d_lr_mult 2.0) — shape fixed, precision now binds

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `parameter_change`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/dtrack2/README.md) · [Record](records/history-context-lrfree-dtrack2-3422a7550929.json)

### context:lrfree/evidence · history-context-lrfree-evidence-e3fd3da87d09

**Scope:** historical; family_context; revision `c6f204735300d3d5a9523d61a3126784b87a89fd2e89e8bfbc9de352ebfc0293`.

AMSGrad on DV12 (PR202 port): results and diagnosis

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/evidence/amsgrad-dv12.md) · [Record](records/history-context-lrfree-evidence-e3fd3da87d09.json)

### context:lrfree/harness · history-context-lrfree-harness-1b5b6bc1e623

**Scope:** historical; family_context; revision `34ba4a239ad791436612df895ca9a4cb2c8c6555cc05bdb39915951f17d2e427`.

lrfree screening harness + GPU pool (2026-09-26)

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/harness/README.md) · [Record](records/history-context-lrfree-harness-1b5b6bc1e623.json)

### context:lrfree/mmd-witness-preflight · history-context-lrfree-mmd-witness-preflight-475f7a8acd27

**Scope:** historical; family_context; revision `45bdc54b0f992061265198e17d5b664c2f4cf5adbd6353e79970de485597ab78`.

Read-only MMD witness preflight on critic-floor native100 states

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/mmd-witness-preflight/README.md) · [Record](records/history-context-lrfree-mmd-witness-preflight-475f7a8acd27.json)

### context:lrfree/overview · history-context-lrfree-overview-df2d587ad5c7

**Scope:** historical; family_context; revision `563db19d2df865dcf8df58df5f045b022afd3f28b731094d99313e6f07629cf7`.

LR-free GAN base search — September 27

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/README.md) · [Record](records/history-context-lrfree-overview-df2d587ad5c7.json)

### context:lrfree/paired-bd-graft · history-context-lrfree-paired-bd-graft-2a88f7c9f74a

**Scope:** historical; family_context; revision `2bbc5594692f88e508e1644c0cd114512ad170871f0e83e41a4117666a65e12c`.

Best measured native lead: paired birth/death on the sigma-floor candidate

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/paired-bd-graft/README.md) · [Record](records/history-context-lrfree-paired-bd-graft-2a88f7c9f74a.json)

### context:lrfree/prior-geomean · history-context-lrfree-prior-geomean-ee68d53e75bc

**Scope:** historical; family_context; revision `b9e2f378e9d7fb4c4a75e673d1c7100c038bbad91110d3b584c8492b7068ae9b`.

Geometric prior-rate coupling: grid100 holdout pass, frozen run fail

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-geomean/README.md) · [Record](records/history-context-lrfree-prior-geomean-ee68d53e75bc.json)

### context:lrfree/prior-rate-coupling · history-context-lrfree-prior-rate-coupling-57cfb10e6b76

**Scope:** historical; family_context; revision `eead5f92c1f30c51767eda8c10030d02a552bd9085d1bb56ab77c7ada4bca044`.

Prior-rate coupling native100 near miss

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-rate-coupling/README.md) · [Record](records/history-context-lrfree-prior-rate-coupling-57cfb10e6b76.json)

### context:lrfree/prior-settle-handoff · history-context-lrfree-prior-settle-handoff-9bc5f8db65e4

**Scope:** historical; family_context; revision `3028029c2e168e7c587d3200739070c1af22c5506d27436d236e12d2f6b62d90`.

First frozen native grid100 PASS: state-driven prior-rate handoff

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-settle-handoff/README.md) · [Record](records/history-context-lrfree-prior-settle-handoff-9bc5f8db65e4.json)

### context:lrfree/row-em-renew · history-context-lrfree-row-em-renew-95b6bd595179

**Scope:** historical; family_context; revision `b315e887b77144d454d52af8a9976efb0d571cbb714658f48f54d583f23f1c24`.

Fresh-reservoir sampling calibration: native100 3/3

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `sampling_only_patch`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/README.md) · [Record](records/history-context-lrfree-row-em-renew-95b6bd595179.json)

### context:lrfree/row-em-sampling · history-context-lrfree-row-em-sampling-cbeebf17f26a

**Scope:** historical; family_context; revision `fbdba5c333161dcabf0a2c03cd52b6ea975362cf1ee2a31aa71bcb4bbf68e671`.

Sampling-only row-mass and width calibration for PR 155

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `sampling_only_patch`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-sampling/README.md) · [Record](records/history-context-lrfree-row-em-sampling-cbeebf17f26a.json)

### context:lrfree/sigdata-bdpair · history-context-lrfree-sigdata-bdpair-b7bf4a720710

**Scope:** historical; family_context; revision `d009775f755c6a49a90a8e6dbfad87e6d3508bf0b0430ba70d1ef8b7006d37e7`.

sigdata-v1 + paired birth/death graft: native100 2/3 (honest generator fix)

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigdata-bdpair/README.md) · [Record](records/history-context-lrfree-sigdata-bdpair-b7bf4a720710.json)

### context:lrfree/sigma-adaptation-audit · history-context-lrfree-sigma-adaptation-audit-21c148e4b6ce

**Scope:** historical; family_context; revision `0d4e02a22fdbb7c017c23f417d90ab454a4dd3b34ab7d766ecef9b33f91be21e`.

Effective sigma release and shared-clock audit (native grid100)

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigma-adaptation-audit/README.md) · [Record](records/history-context-lrfree-sigma-adaptation-audit-21c148e4b6ce.json)

### context:lrfree/structural100 · history-context-lrfree-structural100-8da5accab480

**Scope:** historical; family_context; revision `45ff7a6b599787118987a3c527f1b291d6a17597e6b6c0e03048b60cfd77ebb8`.

Native 100-Gaussian structural round (`st5`–`st7` and D-rate ablation)

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `training_formulation`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/README.md) · [Record](records/history-context-lrfree-structural100-8da5accab480.json)

### context:report/behavioral_baseline · history-context-report-behavioral-baseline-e67454a38f2c

**Scope:** historical; family_context; revision `ee986dd57fad064e7111ff58f069ec777a0202a8c994a561ac0efa0ea7d44920`.

Behavioral baseline — live weights

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/behavioral_baseline/README.md) · [Record](records/history-context-report-behavioral-baseline-e67454a38f2c.json)

### context:report/cifar-ddgan · history-context-report-cifar-ddgan-ceab3dcc1ee7

**Scope:** historical; family_context; revision `f78d1dd141c6e1afeae229e4d9cded89de4d70b3381f45bf00d66a7b8785d33b`.

CIFAR-10 U-Net DDGAN baseline

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/cifar-ddgan/README.md) · [Record](records/history-context-report-cifar-ddgan-ceab3dcc1ee7.json)

### context:report/cifar-particle-ae · history-context-report-cifar-particle-ae-f33e77856a0f

**Scope:** historical; family_context; revision `1279c70e3896f85ed5b22a2494b2dfdfa91efb726edd363335b0d09683968a8a`.

Direct CIFAR particle autoencoder: first matched baseline

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/cifar-particle-ae/README.md) · [Record](records/history-context-report-cifar-particle-ae-f33e77856a0f.json)

### context:report/cifar-particle-ddgan · history-context-report-cifar-particle-ddgan-20d9cb478e20

**Scope:** historical; family_context; revision `781acdeb22b24979d8573b086a066187b47c5cad89ab09cf0cad5e08f3d4e273`.

Direct particle AE-GAN versus DDGAN at 10k updates

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/cifar-particle-ddgan/README.md) · [Record](records/history-context-report-cifar-particle-ddgan-20d9cb478e20.json)

### context:report/denoising-toy · history-context-report-denoising-toy-b99cf0511211

**Scope:** historical; family_context; revision `50bfc2f41e796c0d1284523dfccc96e287d9d5e4d6a6ce31ccf9bd85d5162362`.

Benchmark: conditional denoising on 100 Gaussians

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/denoising-toy/README.md) · [Record](records/history-context-report-denoising-toy-b99cf0511211.json)

### context:report/gym · history-context-report-gym-98d7f8d0cc29

**Scope:** historical; family_context; revision `97b67c8f89a835b94891407e577e27e243c55dcac5217e2974ae82384c6f4fe1`.

Lunar Lander baseline leaderboard

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/gym/lunar_lander/baseline/README.md) · [Record](records/history-context-report-gym-98d7f8d0cc29.json)

### context:report/learned_lr · history-context-report-learned-lr-8aa9c1257d65

**Scope:** historical; family_context; revision `cbc708d0193283a43bff8738926259c30778adfcf7315f8c7fafab3c09a2c34e`.

Learned GAN LR controller — initial study

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/learned_lr/README.md) · [Record](records/history-context-report-learned-lr-8aa9c1257d65.json)

### context:report/locked_shared · history-context-report-locked-shared-5986fd28b1ad

**Scope:** historical; family_context; revision `01daf2649bd1b6c592c22f07774baa84c32cc3705df93736d1d0d3542ee57cdf`.

Locked shared: measured behavior

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/locked_shared/README.md) · [Record](records/history-context-report-locked-shared-5986fd28b1ad.json)

### context:report/lunar_fast · history-context-report-lunar-fast-6bc3bcaf31fa

**Scope:** historical; family_context; revision `0b9dd831e98298bdd4ef05636589bbc2a3014e02f3c17bb46b6a5a253c0f5e2c`.

Fast Lunar lander: full command evidence

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/lunar_fast/README.md) · [Record](records/history-context-report-lunar-fast-6bc3bcaf31fa.json)

### context:report/mog-autoencoder · history-context-report-mog-autoencoder-2c12cda79abc

**Scope:** historical; family_context; revision `f955fcf35ba8326c8909aa9a548a78b3fb521376e98a7d9da621dd84be08dcfa`.

Fixed-sigma particle routing on 100 Gaussians

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/mog-autoencoder/README.md) · [Record](records/history-context-report-mog-autoencoder-2c12cda79abc.json)

### context:report/mog-vae · history-context-report-mog-vae-206767d0b95e

**Scope:** historical; family_context; revision `9271e22327cb5565f09e29698cf118108452b35b3016ce28d14ac380ed815878`.

Particle VAE toy experiments

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/mog-vae/README.md) · [Record](records/history-context-report-mog-vae-206767d0b95e.json)

### context:report/paired_error_2d · history-context-report-paired-error-2d-16f88869c905

**Scope:** historical; family_context; revision `b3a30a125bd8fa50fab16c61f4ddeb5c3c0b28ded03e95d54249abd3f8a4c351`.

Paired 2D transport: MSE-free cap/schedule transfer

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/paired_error_2d/README.md) · [Record](records/history-context-report-paired-error-2d-16f88869c905.json)

### context:report/prior-comparison · history-context-report-prior-comparison-678db27f3d00

**Scope:** historical; family_context; revision `d68f0cb3ca25af16629127a2783debd12bb7c20483ef8d3adfe5c0ba9c08064a`.

Matched prior comparison: 100 Gaussians

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/prior-comparison/README.md) · [Record](records/history-context-report-prior-comparison-678db27f3d00.json)

### context:report/project · history-context-report-project-81e190acdb06

**Scope:** historical; family_context; revision `31d37f13772254c2735762a593936b81158bd50d9b17722eab708394d8cadcbb`.

Research reports and local artifacts

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/README.md) · [Record](records/history-context-report-project-81e190acdb06.json)

### context:report/readme-100gaussians · history-context-report-readme-100gaussians-146d5f77c735

**Scope:** historical; family_context; revision `4a159041e19609cf526c51b6cd2c1478d0611cc7ba9e1aa3bbc9c7657df5e9e8`.

README 100-Gaussian visualization

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/readme-100gaussians/README.md) · [Record](records/history-context-report-readme-100gaussians-146d5f77c735.json)

### context:report/releases · history-context-report-releases-ad7e3c09bcd5

**Scope:** historical; family_context; revision `b8ce9c6f8a9018e035073b4c5b2f32e64ada07575758ece01a4c97313e607d61`.

0.5.0 release validation

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/releases/0.5.0.md) · [Record](records/history-context-report-releases-ad7e3c09bcd5.json)

### context:report/smart_descent · history-context-report-smart-descent-d18c56c9f363

**Scope:** historical; family_context; revision `76f06a46d1433061e2bbc21df3580e9f52b33dd02cb8ed613b6745964250f78b`.

Smart descent — expanded experiments

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/smart_descent/README.md) · [Record](records/history-context-report-smart-descent-d18c56c9f363.json)

### context:report/sparse-ucd · history-context-report-sparse-ucd-8f8233db045d

**Scope:** historical; family_context; revision `c71381249dd4a6b4082688809041593409d0722c331a1313388371997fae9910`.

Sparse conditional mixed-output study (UCD) — protocol

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/sparse-ucd/README.md) · [Record](records/history-context-report-sparse-ucd-8f8233db045d.json)

### context:report/toy100/accuracy-failure · history-context-report-toy100-accuracy-failure-f7fbb99ec969

**Scope:** historical; family_context; revision `e8c8fa6bbf5dd3e99e6952058e6ce69fb79587cb2b529d1c1d4e4cd0f28db08a`.

Early full coverage followed by collapse

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/accuracy-failure/README.md) · [Record](records/history-context-report-toy100-accuracy-failure-f7fbb99ec969.json)

### context:report/toy100/accuracy-horizon · history-context-report-toy100-accuracy-horizon-7be38bd4da8f

**Scope:** historical; family_context; revision `2c082f0031c0c201f111d3deddc4812d2a83d20a192fc22e9e57d0bdc9b19aae`.

One capped schedule: two accurate 100-mode cases, one failure

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/accuracy-horizon/README.md) · [Record](records/history-context-report-toy100-accuracy-horizon-7be38bd4da8f.json)

### context:report/toy100/accuracy-network-horizon · history-context-report-toy100-accuracy-network-horizon-bcb2c0133162

**Scope:** historical; family_context; revision `0e0b9af1e9617c5f972a94322a3a8e7dd3ca4ee9777872a34ba767051ffab7c9`.

One experimental configuration passes all three 100-mode accuracy gates

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/accuracy-network-horizon/README.md) · [Record](records/history-context-report-toy100-accuracy-network-horizon-bcb2c0133162.json)

### context:report/toy100/baseline · history-context-report-toy100-baseline-40b7923a3162

**Scope:** historical; family_context; revision `fec8bce95abe93efb0682cf7fa49c0baad80eed2cd5dd10ce67585e3f7733d77`.

100-Gaussian toy gate: FAIL

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/baseline/leaderboard.md) · [Record](records/history-context-report-toy100-baseline-40b7923a3162.json)

### context:report/toy100/batch-feature-init · history-context-report-toy100-batch-feature-init-2a6c7e903d7b

**Scope:** historical; family_context; revision `bdd2f043deab044d3df57e4d0b314ce6942ceb92824b1c9a0d5282a44c12397c`.

Batch-feature initialization: research and package integration

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/batch-feature-init/README.md) · [Record](records/history-context-report-toy100-batch-feature-init-2a6c7e903d7b.json)

### context:report/toy100/cpu-recipe-gpu-port · history-context-report-toy100-cpu-recipe-gpu-port-ff3bab8bd767

**Scope:** historical; family_context; revision `b682c89a5b30fddaacd03585b2b9112829811fd40e39d56d4f0ed90908c98cd0`.

Port the known CPU winner to GPU

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/cpu-recipe-gpu-port/README.md) · [Record](records/history-context-report-toy100-cpu-recipe-gpu-port-ff3bab8bd767.json)

### context:report/toy100/critic_signal_attempt · history-context-report-toy100-critic-signal-attempt-84c41c02a33e

**Scope:** historical; family_context; revision `ffab87b90e4370c08c92b408e6ab52b806688f97f2b5510b576a03604fe27312`.

Constant-rate GAN acquisition leader H

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/critic_signal_attempt/README.md) · [Record](records/history-context-report-toy100-critic-signal-attempt-84c41c02a33e.json)

### context:report/toy100/dimension-rms-base · history-context-report-toy100-dimension-rms-base-c4dbe5b2c3a1

**Scope:** historical; family_context; revision `ea6cabd4dd9cc62cf9f9a462aab0e3b225a7a6a717ca2890b190d9a9349ac776`.

Selected research base: dimension RMS hybrid

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/dimension-rms-base/README.md) · [Record](records/history-context-report-toy100-dimension-rms-base-c4dbe5b2c3a1.json)

### context:report/toy100/direct-particle-base · history-context-report-toy100-direct-particle-base-dedd9745d572

**Scope:** historical; family_context; revision `aa2e884b643dc01ab678cd4b726fcba146264e8aff3b6eace3ce59e82e1868dc`.

Previous research base: direct_particle_response

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/direct-particle-base/README.md) · [Record](records/history-context-report-toy100-direct-particle-base-dedd9745d572.json)

### context:report/toy100 · history-context-report-toy100-eab3beb766f7

**Scope:** historical; family_context; revision `667f7130dfd21da2ebbd039333e6d78927798790bdbb9074f69d7636bd2b2bf2`.

100-mode convergence: failures and shared-recipe search

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/README.md) · [Record](records/history-context-report-toy100-eab3beb766f7.json)

### context:report/toy100/formulation-round-20260924 · history-context-report-toy100-formulation-round-20260924-7c545c79a8d0

**Scope:** historical; family_context; revision `9ceec802c64f63b9dd3f48dbd902ffde768545c24477d655874b57f9cb835f67`.

Formulation round: 18 proposals, no joint winner

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/formulation-round-20260924/README.md) · [Record](records/history-context-report-toy100-formulation-round-20260924-7c545c79a8d0.json)

### context:report/toy100/gap-fill-20260925 · history-context-report-toy100-gap-fill-20260925-2f29e447871c

**Scope:** historical; family_context; revision `74ce5d3a3a3e1142c760ba1d954f56cbb3f5b62f09648c0fe8a0756a6efd3c03`.

Qualification gap fill — September 25, 2026

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/README.md) · [Record](records/history-context-report-toy100-gap-fill-20260925-2f29e447871c.json)

### context:report/toy100/gpu-known-winner-control · history-context-report-toy100-gpu-known-winner-control-24dd3e170c95

**Scope:** historical; family_context; revision `32d081a956a8660d910d97e728d372e43c138ce1e964e060134140dfe6bb54e4`.

Previously passing 22-toy recipe: CPU and GPU control

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gpu-known-winner-control/README.md) · [Record](records/history-context-report-toy100-gpu-known-winner-control-24dd3e170c95.json)

### context:report/toy100/gpu-leaderboard · history-context-report-toy100-gpu-leaderboard-ddf28349e721

**Scope:** historical; family_context; revision `5ff6aca63294f3eb862c831bf999f26d933e653d71520d3abd7719fc29a593c2`.

GPU leaderboard rebuild

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gpu-leaderboard/README.md) · [Record](records/history-context-report-toy100-gpu-leaderboard-ddf28349e721.json)

### context:report/toy100/h_stability · history-context-report-toy100-h-stability-1a6179c5bdbd

**Scope:** historical; family_context; revision `7fd68a8a4f8949a93a18fcabf4b133aac3ff6caa74d3282a9a092fc2706fb0aa`.

Current experiment base

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/h_stability/README.md) · [Record](records/history-context-report-toy100-h-stability-1a6179c5bdbd.json)

### context:report/toy100/k3p-base · history-context-report-toy100-k3p-base-537d23f0d17c

**Scope:** historical; family_context; revision `bd0ea201ac1adb5efd07991e668c7c70148cb0e6d93311bf4c7f33eab908812d`.

Selected research base: K3P

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/k3p-base/README.md) · [Record](records/history-context-report-toy100-k3p-base-537d23f0d17c.json)

### context:report/toy100/overnight-20260925 · history-context-report-toy100-overnight-20260925-2148232a78ce

**Scope:** historical; family_context; revision `982c7ee41777e58ce31c7ddb58ecfeacf9f3fafecb81d4b311436c41f1658b16`.

Completed overnight evidence — September 25, 2026

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/overnight-20260925/README.md) · [Record](records/history-context-report-toy100-overnight-20260925-2148232a78ce.json)

### context:report/toy100/pr140-pr143-repro-audit · history-context-report-toy100-pr140-pr143-repro-audit-06445544e4db

**Scope:** historical; family_context; revision `f96a130d45ca38a9ecc946f750767fd241c40d0ad8dc2425ea1e56cbd4399ea4`.

PR140 / PR143 CPU reproduction audit

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/pr140-pr143-repro-audit/README.md) · [Record](records/history-context-report-toy100-pr140-pr143-repro-audit-06445544e4db.json)

### context:report/toy100/recommended · history-context-report-toy100-recommended-10cf631bab45

**Scope:** historical; family_context; revision `b28370006c6192d8ce844ddc0d2a0adcf0829a54a73f9957cedaf32a80c4ebbd`.

100-Gaussian toy gate: PASS

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/recommended/leaderboard.md) · [Record](records/history-context-report-toy100-recommended-10cf631bab45.json)

### context:report/toy100/search · history-context-report-toy100-search-008d0b30e272

**Scope:** historical; family_context; revision `7cdfa3072b1fdebd97ff674e9dffb88a42588bab849604ceb9e7fc7687bd36f4`.

Configuration search ledger

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/search/README.md) · [Record](records/history-context-report-toy100-search-008d0b30e272.json)

### context:report/toy100/shared22 · history-context-report-toy100-shared22-d31ae191ca97

**Scope:** historical; family_context; revision `063fd99f767735aa002fc47bdc70a3c6956180f69adc819e07e6f4856ff0f865`.

One shared recipe: 22/22

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/shared22/README.md) · [Record](records/history-context-report-toy100-shared22-d31ae191ca97.json)

### context:report/toy100/simpler22 · history-context-report-toy100-simpler22-e2f8ba01e6a2

**Scope:** historical; family_context; revision `8da434df22444c6ce6d810e9b2c0fc444c467380e1b7a682452e7106a8891414`.

Simpler shared recipe: 22/22

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/simpler22/README.md) · [Record](records/history-context-report-toy100-simpler22-e2f8ba01e6a2.json)

### context:report/toy100/torch214-gpu-check · history-context-report-toy100-torch214-gpu-check-fbe95d378065

**Scope:** historical; family_context; revision `ddb6c00dae64399f4c9ab0e30fd61a963929a63a88fd39a17becbc7be30a02de`.

PyTorch 2.14 does not fix the two GPU blockers

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/torch214-gpu-check/README.md) · [Record](records/history-context-report-toy100-torch214-gpu-check-fbe95d378065.json)

### context:report/trajectory · history-context-report-trajectory-79bf115a46be

**Scope:** historical; family_context; revision `791a3c868e45dc8bc4645a8aa81b11193f17f169941432c2fa45394fe3aeb260`.

Conditional trajectories with Particle DDGAN

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/trajectory/README.md) · [Record](records/history-context-report-trajectory-79bf115a46be.json)

### context:report/transfer_suite · history-context-report-transfer-suite-06e9f9952ffa

**Scope:** historical; family_context; revision `ce617fc5bd68c3395ef4a45abdbfeb9cdd8ab0784f22fdaad7b299b0b7d3f5f0`.

Transfer leaderboard with test importance

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/transfer_suite/README.md) · [Record](records/history-context-report-transfer-suite-06e9f9952ffa.json)

### context:report/transfer_suite/anisotropic_core_metric · history-context-report-transfer-suite-anisotropic-core-metric-cfb4b9ab9711

**Scope:** historical; family_context; revision `925a5e5ac53fc79b133a0df9f873196265c5c880f0f38bc0d4e496fca10627c6`.

Anisotropic core metric and SiLU critic screen

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/a8b9d3977701ca700d9918ac66d40ac814b9f9ba/reports/transfer_suite/anisotropic_core_metric/README.md) · [Record](records/history-context-report-transfer-suite-anisotropic-core-metric-cfb4b9ab9711.json)

### context:report/transfer_suite/silu_rare_collapse · history-context-report-transfer-suite-silu-rare-collapse-fbfea951783c

**Scope:** historical; family_context; revision `a45cc1de416e2f5a0bf403cf6057c8f15c42d67f0651b1b00758d2d1493d3f9f`.

Why axis_silu collapses the rare component on vector_unequal_mass

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/a8b9d3977701ca700d9918ac66d40ac814b9f9ba/reports/transfer_suite/silu_rare_collapse/README.md) · [Record](records/history-context-report-transfer-suite-silu-rare-collapse-fbfea951783c.json)

### context:report/transition · history-context-report-transition-61b143595603

**Scope:** historical; family_context; revision `8a0f4df68ff73f5d664afdd22dd3eaac9d51ae1783a722cc54d21995570e8ab3`.

Single-transition GAN results

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `unknown`.



Narrative context only; no scientific pass is inferred.

**Next:** Read the linked narrative and normalize any remaining exact experiment identities before scientific comparison.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/transition/README.md) · [Record](records/history-context-report-transition-61b143595603.json)

### dv12-ams-rc3-c22 · history-dv12-ams-rc3-c22-501063495713

**Scope:** historical; scientific; revision `aef86862b376cb681be9d48ba344f5c3f004a1e005a2e697ec544a9bf3b1de04`.

dv12-ams-rc3 (#217 default) on the 8 custom hosts (custom22 port)

**Observed:** {'FAIL': 4, 'PASS': 4}; wall seconds unknown; mechanism `unknown`.

ae_gan_hold: recon_mse=0.0036909

Historical recorded task outcomes: FAIL=4, PASS=4

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-dv12-ams-rc3-c22-501063495713.json)

### h2-handoff-critic-floor · history-h2-handoff-critic-floor-2217ce39ca7d

**Scope:** historical; scientific; revision `5af191815a44b854369297d4862ed50be0e5625a3b1a96c4cefc1981d67aac36`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 353.87; mechanism `training_formulation`.

grid100: modes=100, hq=0.9799, precision=0.9799, acc_center_rms_sigma=0.14004, mass_tv=0.03325, acc_radial_ks=0.028503

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/critic-floor/grid100-result.json) · [Record](records/history-h2-handoff-critic-floor-2217ce39ca7d.json)

### h2-handoff-critic-floor · history-h2-handoff-critic-floor-3ff4d40cea8a

**Scope:** historical; scientific; revision `c4ba18be75d5cd7f78c9feec8d7da5f4cdc7bdc3b5ae863843947df0708d156b`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 283.13; mechanism `training_formulation`.

staggered100: modes=100, hq=0.9864, precision=0.9864, acc_center_rms_sigma=0.16299, mass_tv=0.0366, acc_radial_ks=0.043219

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/critic-floor/staggered100-result.json) · [Record](records/history-h2-handoff-critic-floor-3ff4d40cea8a.json)

### h2-handoff-critic-floor · history-h2-handoff-critic-floor-dbc3b124beec

**Scope:** historical; scientific; revision `a791d502aa9509f61e0984304637ebf31bd8b78699af5d3e58270ec58085fc37`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 347.7; mechanism `training_formulation`.

rotated100: modes=100, hq=0.9526, precision=0.9526, acc_center_rms_sigma=0.16671, mass_tv=0.0417, acc_radial_ks=0.10324

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/critic-floor/rotated100-result.json) · [Record](records/history-h2-handoff-critic-floor-dbc3b124beec.json)

### h2-prior-couple · history-h2-prior-couple-23f283338237

**Scope:** historical; scientific; revision `72adbf6abbca9f19e46df1dcfb9a8f1593b46b7f8fae9367ca65626bce52e7b8`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 419.3; mechanism `unknown`.

grid100: modes=100, hq=0.96875, precision=0.96875, acc_center_rms_sigma=0.12646, mass_tv=0.0337, acc_radial_ks=0.032984

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-rate-coupling/grid100-result.json) · [Record](records/history-h2-prior-couple-23f283338237.json)

### h2-prior-couple-sigma · history-h2-prior-couple-sigma-35033abf8d31

**Scope:** historical; scientific; revision `4e1ea3b6a52c97cb003583bd24f4b92d15eac5d88d92f7ec801c2bd6d2c637e9`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 368.92; mechanism `unknown`.

grid100: modes=100, hq=0.98425, precision=0.98425, acc_center_rms_sigma=0.12048, mass_tv=0.0373, acc_radial_ks=0.10117

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigma-adaptation-audit/fast-result.json) · [Record](records/history-h2-prior-couple-sigma-35033abf8d31.json)

### h2-prior-couple-sigma-clock · history-h2-prior-couple-sigma-clock-6e351f9ddf09

**Scope:** historical; scientific; revision `8c47fd3b7310e8c11279c63ba968c18e941fa3e429f8520ea8d834bd7af26706`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 358.47; mechanism `unknown`.

grid100: modes=100, hq=0.97065, precision=0.97065, acc_center_rms_sigma=0.12402, mass_tv=0.0338, acc_radial_ks=0.044008

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/sigma-adaptation-audit/clock-result.json) · [Record](records/history-h2-prior-couple-sigma-clock-6e351f9ddf09.json)

### h2-prior-geomean · history-h2-prior-geomean-7450032bf52a

**Scope:** historical; scientific; revision `03b30801c16595652862a58a634dd75fc4a32b97bc1c32461e90a689a5b9c3d3`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 290.99; mechanism `unknown`.

grid100: modes=100, hq=0.9799, precision=0.9799, acc_center_rms_sigma=0.20427, mass_tv=0.0364, acc_radial_ks=0.011676

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-geomean/grid100-result.json) · [Record](records/history-h2-prior-geomean-7450032bf52a.json)

### h2-prior-handoff · history-h2-prior-handoff-62ec131e0dcd

**Scope:** historical; scientific; revision `37900df825c572524cd3f822b12699f999ab9dfe5adc3a5c7d287a75de8f0268`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 358.45; mechanism `unknown`.

rotated100: modes=100, hq=0.94155, precision=0.94155, acc_center_rms_sigma=0.24464, mass_tv=0.04665, acc_radial_ks=0.11513

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-settle-handoff/rotated100-result.json) · [Record](records/history-h2-prior-handoff-62ec131e0dcd.json)

### h2-prior-handoff · history-h2-prior-handoff-d50d24befb2b

**Scope:** historical; scientific; revision `73665159226dee4201a5c3d74ec3c4ddccc908a98a64f8495f59d2a5b9ca5c0c`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 288.57; mechanism `unknown`.

staggered100: modes=100, hq=0.9865, precision=0.9865, acc_center_rms_sigma=0.17282, mass_tv=0.0362, acc_radial_ks=0.040299

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-settle-handoff/staggered100-result.json) · [Record](records/history-h2-prior-handoff-d50d24befb2b.json)

### h2-prior-handoff · history-h2-prior-handoff-ddae8c7fb5bc

**Scope:** historical; scientific; revision `9292179e0d33d9e62c249fa7a2f8debbbf156ca73a17434b532a4f3320ba9eb1`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 365.91; mechanism `unknown`.

grid100: modes=100, hq=0.9799, precision=0.9799, acc_center_rms_sigma=0.14004, mass_tv=0.03325, acc_radial_ks=0.028503

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/prior-settle-handoff/grid100-result.json) · [Record](records/history-h2-prior-handoff-ddae8c7fb5bc.json)

### k3 · history-k3-0a4a08142ffd

**Scope:** historical; scientific; revision `b3ad745e52ad571a927ce711313b814118cb79f6f66364dcd74c6d682bd61c2b`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 73.144; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3-0a4a08142ffd.json)

### k3 · history-k3-147811876474

**Scope:** historical; scientific; revision `c94fca4b9e0ec46647cde7e5d784eca93b4ccd13be9eada55b7bb5d0173f1e96`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 107.62; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1, PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3-147811876474.json)

### k3 · history-k3-87d4136d53dc

**Scope:** historical; scientific; revision `2613f5576ee64d6b3390f747af257fe19e7b661b00af68b7e0328b0acdb58bcb`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 112.654; mechanism `unknown`.

mode_hold:shift: modes=8, hq=0.89526

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3-87d4136d53dc.json)

### k3 · history-k3-8f678149295a

**Scope:** historical; scientific; revision `2545387aaded0601a1152f6c93b42b62247f339b702a2646c8b536afa7814299`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 22}; wall seconds unknown; mechanism `training_formulation`.

ae_gan_hold: recon_mse=0.017286; img_bars4: modes=4, hq=0.96875

Historical recorded task outcomes: PASS=22

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/qualification-summary.json) · [Record](records/history-k3-8f678149295a.json)

### k3 · history-k3-96ffa6c63f50

**Scope:** historical; scientific; revision `1ffe66b6d1890d4a77a2240636ca6a20b00e2d03f8da6de4c4fa099f5c46c3bf`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 75.623; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3-96ffa6c63f50.json)

### k3 · history-k3-be723a2faab7

**Scope:** historical; scientific; revision `26f8463332da694b356d82da96b34894d2fc0b0d52945a57eede349c44bc507e`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 70.22; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3-be723a2faab7.json)

### k3g · history-k3g-1185b5a3e9e3

**Scope:** historical; scientific; revision `61c83fb3eec71ed698701ac9f8e30a6ba80922ded01402b59b4cfdfd16b5373c`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 160.482; mechanism `unknown`.

grid100: modes=100, hq=0.98905, precision=0.98905, mass_tv=0.0412

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-1185b5a3e9e3.json)

### k3g · history-k3g-27ec6750db46

**Scope:** historical; scientific; revision `6a2c31fa8e413c17a4cf97a290c0f5a477bad11dfb12e2ce4fa8d074ad87d29c`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 22}; wall seconds unknown; mechanism `training_formulation`.

ae_gan_hold: recon_mse=0.017286; img_bars4: modes=4, hq=0.96875

Historical recorded task outcomes: PASS=22

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/qualification-summary.json) · [Record](records/history-k3g-27ec6750db46.json)

### k3g · history-k3g-7a718fd73463

**Scope:** historical; scientific; revision `d26407f43842ffdcaa1c394e8b79bd9c6d31a22d9652c70823d47ecb4427b154`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 215.751; mechanism `unknown`.

mode_hold:shift: modes=8, hq=0.89526

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-7a718fd73463.json)

### k3g · history-k3g-a79df20a0f54

**Scope:** historical; scientific; revision `3187d2d8d8a049264bdadcb132437827cc240c041e2d50dfbe208fa819050497`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 204.649; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1, PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-a79df20a0f54.json)

### k3g · history-k3g-a7a61862a334

**Scope:** historical; scientific; revision `f1c2913ff6ba40a1fcad8d3043dd539613912858d201218ece0eebe7e98739ca`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 104.189; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-a7a61862a334.json)

### k3g · history-k3g-c0741e2a3be4

**Scope:** historical; scientific; revision `9667f72d9a497a004c5d848bd73b8acf4fffec325358da9d2779a298b907c6cf`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 223.513; mechanism `unknown`.

rotated100: modes=100, hq=0.98345, precision=0.98345, mass_tv=0.0353

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-c0741e2a3be4.json)

### k3g · history-k3g-ef4a34363603

**Scope:** historical; scientific; revision `a070c4a80818f44df9aa0a1d60969d3a271c3d649840613adabb00b5efb5a43d`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 66.428; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-ef4a34363603.json)

### k3g · history-k3g-f710ebdb0a06

**Scope:** historical; scientific; revision `4efc458cd68b748d9add02642664d4d3fa3f8f160fce1062bf0699baa8a72301`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 73.545; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3g-f710ebdb0a06.json)

### k3p · history-k3p-0b8eb506eb0e

**Scope:** historical; scientific; revision `70e613b0991df1a5fa8891904632899dedefb5ff2091eac7769ddcca4eee48b0`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 24.307; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-0b8eb506eb0e.json)

### k3p · history-k3p-1d9062e41064

**Scope:** historical; scientific; revision `b1e6e1c877295ccee64e718e7583956525cfebd26fc2a87484bd56793a170500`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 75.404; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-1d9062e41064.json)

### k3p · history-k3p-228a849bf4a7

**Scope:** historical; scientific; revision `abbdc2700ef6b20360eb2aab169c53931267e50acc56419825756588a9b6f3e1`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 16.067; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-228a849bf4a7.json)

### k3p · history-k3p-4ed085f0b363

**Scope:** historical; scientific; revision `37347e3dcbb1800a88cf34df606ed85af024b3c5f31dc6073a366b310c4aab51`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 24.136; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-4ed085f0b363.json)

### k3p · history-k3p-60f893f86276

**Scope:** historical; scientific; revision `a567ca6dc5b5b27a61cad5224fb47301466b70735c9ca718f1e51fca265f6f56`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 5.754; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-60f893f86276.json)

### k3p · history-k3p-7f1a8178bed4

**Scope:** historical; scientific; revision `2274083a36e70b8acb694ff9a6470045ac11617e7c8530b7cb8af12f360366ac`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 209.731; mechanism `unknown`.

staggered100: modes=100, hq=0.98675, precision=0.98675, mass_tv=0.0419

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-7f1a8178bed4.json)

### k3p · history-k3p-84363f69cb3a

**Scope:** historical; scientific; revision `b30920c7ba6b00eb9c04879efa6027425c02310877c70c7bffb9a3591768afea`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 22}; wall seconds unknown; mechanism `training_formulation`.

ae_gan_hold: recon_mse=0.017188; img_bars4: modes=4, hq=0.96875

Historical recorded task outcomes: PASS=22

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/qualification-summary.json) · [Record](records/history-k3p-84363f69cb3a.json)

### k3p · history-k3p-893c76d9b17d

**Scope:** historical; scientific; revision `d171c8a42006436bde3ec09a387615835ca7fac6e819bc01eab3cafc700435ce`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 84.475; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-893c76d9b17d.json)

### k3p · history-k3p-93c25177b3fd

**Scope:** historical; scientific; revision `100aa95b6fef13c284209498da920c3754473fba9fc5d98a109990035dc9beed`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 160.566; mechanism `unknown`.

rotated100: modes=100, hq=0.98175, precision=0.98175, mass_tv=0.036

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-93c25177b3fd.json)

### k3p · history-k3p-9525caf87f89

**Scope:** historical; scientific; revision `059c462672891d6d5ca3a71e3e203149a66a85c21a3bdbca79b90b83bf759e6a`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 78.766; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-9525caf87f89.json)

### k3p · history-k3p-ad80a9e5c460

**Scope:** historical; scientific; revision `de2f6e51ca273a1c0b4eda4dcd6344c415770245e5dce06bbc59ebf92c4b94e9`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 10.089; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-ad80a9e5c460.json)

### k3p · history-k3p-af5b9867d53f

**Scope:** historical; scientific; revision `add962b4d0bedb71deb050640538fc3a99994a42e66ddb399a53e4c59aa1e919`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 55.666; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-af5b9867d53f.json)

### k3p · history-k3p-b85d094063c3

**Scope:** historical; scientific; revision `35bd8469c341c547a471ad677a7916920373c24893531b7d194f10b80f99f2e5`.

Selected K3P stationary endurance and target-shift recovery.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds unknown; mechanism `unknown`.

mode_hold:ring_hold: passing_checks=1200; mode_hold:ring_extension: passing_checks=300

Historical recorded task outcomes: FAIL=1, PASS=2

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/current-research-base.json) · [Record](records/history-k3p-b85d094063c3.json)

### k3p · history-k3p-bd8d9c0fe9ae

**Scope:** historical; scientific; revision `3b9bd4554d652718a776d388d01d83557ea6626db2568305c966a64c5dbdaa3b`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 62.389; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-bd8d9c0fe9ae.json)

### k3p · history-k3p-cc0457e9f8d1

**Scope:** historical; scientific; revision `89bd53eaab8e26f8ff76aa35fecbd28ba2fa6045fabb30fcff62ad8381d32817`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 13.326; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-cc0457e9f8d1.json)

### k3p · history-k3p-d5c6c76bd857

**Scope:** historical; scientific; revision `2e2bfae6c7dfebabb97bc70401304232c5154b0f44ee43371a7a470fcc207dff`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 3.492; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-d5c6c76bd857.json)

### k3p · history-k3p-da4328bf0c90

**Scope:** historical; scientific; revision `632180aac0a3e3076d90f8db0fd01fcf0ceec966a2d76080d9c148afac8e0999`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 79.034; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-da4328bf0c90.json)

### k3p · history-k3p-e186ceabc14c

**Scope:** historical; scientific; revision `d1ff4ef104b501053e09bd2fad96957ebae8234c163d4e15428a07498be1cdf2`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 17.633; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-e186ceabc14c.json)

### k3p · history-k3p-eb1183895ce1

**Scope:** historical; scientific; revision `87e0ea7906b4c59c4209ed43e45c2b204fe779b17b78b048264ea0e928812cb7`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 24.51; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-eb1183895ce1.json)

### k3p · history-k3p-f7c3ad0a986e

**Scope:** historical; scientific; revision `dc362ddea33a2643a4247e318789ca1a81bc14f4dcafaf0743c5e1c52780daa2`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 47.01; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-k3p-f7c3ad0a986e.json)

### k3p-noin-c22-h200 · history-k3p-noin-c22-h200-c51d40f5f659

**Scope:** historical; scientific; revision `1d16500a6c2039ff5821839f2b80879a3c9a197d39b2455ff0084da85e78a494`.

binding control: develop K3P annealed with horizon = host budget 200, host init, critic input noise OFF (engine does not bind input noise)

**Observed:** {'PASS': 1}; wall seconds unknown; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-k3p-noin-c22-h200-c51d40f5f659.json)

### k3p-noin-c22-h250 · history-k3p-noin-c22-h250-497fd4659aef

**Scope:** historical; scientific; revision `a54631269d6ae763c67d8b36fd133efd11314b3580b603c5aba1c4a5c3d7f2e5`.

binding control: develop K3P annealed with horizon = host budget 250, host init, critic input noise OFF (engine does not bind input noise)

**Observed:** {'PASS': 1}; wall seconds unknown; mechanism `unknown`.

ae_gan_hold: recon_mse=0.0087456

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-k3p-noin-c22-h250-497fd4659aef.json)

### k3p-noin-c22-h400 · history-k3p-noin-c22-h400-30abad2abe06

**Scope:** historical; scientific; revision `18d3b1e3d151d2996cd223e3f5c958f55a9eb3cace57b2fca060841a3f61d71b`.

binding control: develop K3P annealed with horizon = host budget 400, host init, critic input noise OFF (engine does not bind input noise)

**Observed:** {'PASS': 3}; wall seconds unknown; mechanism `unknown`.



Historical recorded task outcomes: PASS=3

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-k3p-noin-c22-h400-30abad2abe06.json)

### k3p-noin-c22-h80 · history-k3p-noin-c22-h80-d425775228ba

**Scope:** historical; scientific; revision `aa92e9bd94522fcdd79182bf57cd93fae74e14eeb146038b06cfab83a6c80d33`.

binding control: develop K3P annealed with horizon = host budget 80, host init, critic input noise OFF (engine does not bind input noise)

**Observed:** {'PASS': 1}; wall seconds unknown; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-k3p-noin-c22-h80-d425775228ba.json)

### k3p-noin-c22-h800 · history-k3p-noin-c22-h800-983ed0c7a978

**Scope:** historical; scientific; revision `c2f4bf416468b41a103bb74c8c4f88fa8304b901e39188b67e5e54e3fba63c11`.

binding control: develop K3P annealed with horizon = host budget 800, host init, critic input noise OFF (engine does not bind input noise)

**Observed:** {'PASS': 2}; wall seconds unknown; mechanism `unknown`.



Historical recorded task outcomes: PASS=2

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-k3p-noin-c22-h800-983ed0c7a978.json)

### pr215_exact_host_init · history-pr215-exact-host-init-0ee61e89ff62

**Scope:** historical; scientific; revision `fc26fd5e53d545fce331d99c6d479cda5b6dd5d12c7e34ec7eab40cd04e009d0`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 8, 'PASS': 5}; wall seconds unknown; mechanism `unknown`.

img_bars4: passing_checks=3, observations=24; img_blobs4: passing_checks=12, observations=24; img_intensity2: passing_checks=0, observations=48

Historical recorded task outcomes: FAIL=8, PASS=5

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/results.json) · [Record](records/history-pr215-exact-host-init-0ee61e89ff62.json)

### pr215_exact_host_init · history-pr215-exact-host-init-74bea9b2e76e

**Scope:** historical; scientific; revision `fc26fd5e53d545fce331d99c6d479cda5b6dd5d12c7e34ec7eab40cd04e009d0`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 3}; wall seconds unknown; mechanism `unknown`.

grid100: modes=100, precision=0.9822, acc_center_rms_sigma=0.31956, mass_tv=0.06475, acc_radial_ks=0.015252; rotated100: modes=100, precision=0.96855, acc_center_rms_sigma=0.4429, mass_tv=0.05225, acc_radial_ks=0.03928; staggered100: modes=100, precision=0.97715, acc_center_rms_sigma=0.36379, mass_tv=0.04935, acc_radial_ks=0.037328

Historical recorded task outcomes: FAIL=3

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/results.json) · [Record](records/history-pr215-exact-host-init-74bea9b2e76e.json)

### pr215_qr_adapter · history-pr215-qr-adapter-e0fbccd2b4d8

**Scope:** historical; scientific; revision `331fdbd266382abe38c29a657f98c9ebf4aa7e449d928b99f035bd4fe130bd5c`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 3}; wall seconds unknown; mechanism `unknown`.

grid100: modes=100, precision=0.9857, acc_center_rms_sigma=0.25866, mass_tv=0.05825, acc_radial_ks=0.046864; rotated100: modes=100, precision=0.9669, acc_center_rms_sigma=0.40794, mass_tv=0.0463, acc_radial_ks=0.042986; staggered100: modes=98, precision=0.9783, acc_center_rms_sigma=0.26928, mass_tv=0.08395, acc_radial_ks=0.022997

Historical recorded task outcomes: FAIL=3

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/results.json) · [Record](records/history-pr215-qr-adapter-e0fbccd2b4d8.json)

### pr215_qr_adapter · history-pr215-qr-adapter-e1824439449c

**Scope:** historical; scientific; revision `331fdbd266382abe38c29a657f98c9ebf4aa7e449d928b99f035bd4fe130bd5c`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 6, 'PASS': 7}; wall seconds unknown; mechanism `unknown`.

img_bars4: passing_checks=10, observations=24; img_blobs4: passing_checks=18, observations=24; img_intensity2: passing_checks=36, observations=48

Historical recorded task outcomes: FAIL=6, PASS=7

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/results.json) · [Record](records/history-pr215-qr-adapter-e1824439449c.json)

### pr217_qr_native_adapter · history-pr217-qr-native-adapter-0d320a9304af

**Scope:** historical; scientific; revision `d778b2b9e3c2e430d98693a1c92c8ee3a6f3a6e1f8a55668cf1d9b3c59f33704`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 13}; wall seconds unknown; mechanism `unknown`.

img_bars4: passing_checks=15, observations=24; img_blobs4: passing_checks=17, observations=24; img_intensity2: passing_checks=20, observations=48

Historical recorded task outcomes: PASS=13

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/results.json) · [Record](records/history-pr217-qr-native-adapter-0d320a9304af.json)

### pr217_qr_native_adapter · history-pr217-qr-native-adapter-57bc39c1b2bf

**Scope:** historical; scientific; revision `d778b2b9e3c2e430d98693a1c92c8ee3a6f3a6e1f8a55668cf1d9b3c59f33704`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 3}; wall seconds unknown; mechanism `unknown`.

grid100: modes=99, precision=0.88935, acc_center_rms_sigma=0.37437, mass_tv=0.0663, acc_radial_ks=0.22271; rotated100: modes=100, precision=0.87775, acc_center_rms_sigma=0.22031, mass_tv=0.06385, acc_radial_ks=0.029222; staggered100: modes=100, precision=0.8843, acc_center_rms_sigma=0.33098, mass_tv=0.05075, acc_radial_ks=0.1147

Historical recorded task outcomes: FAIL=3

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/comparison-pr215-pr217/results.json) · [Record](records/history-pr217-qr-native-adapter-57bc39c1b2bf.json)

### rg5-a2 · history-rg5-a2-54276d11a692

**Scope:** historical; scientific; revision `5a7899ee49f5df4b69687e5fac23550ea9a3ebf8759618ccbae6cd2139e0e4b0`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 146.301; mechanism `unknown`.

mode_hold:shift_frozen: modes=0, hq=0

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-a2-54276d11a692.json)

### rg5-a2-floor-0.1-0.1 · history-rg5-a2-floor-0-1-0-1-a61c458edd6f

**Scope:** historical; scientific; revision `e41a7cc0b1647f4a279d8f26b30038f9bf27952bfa056a0929f4c7cf73045d19`.

Matched frozen control distinguishes adaptation from stationary output.

**Observed:** {'PASS': 1}; wall seconds unknown; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/rg5-recovery-pair.json) · [Record](records/history-rg5-a2-floor-0-1-0-1-a61c458edd6f.json)

### rg5-bcap · history-rg5-bcap-136441020f46

**Scope:** historical; scientific; revision `a330a9a2806f69df8ddd0e6cea1720ef31fa8973d010baffac0e8c7d143eb3af`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 2.928; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-136441020f46.json)

### rg5-bcap · history-rg5-bcap-1e8c9727083a

**Scope:** historical; scientific; revision `19caadd3bd025f8f71687b78e484a39235d55f0b7cef0e44725a43c3357dc888`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 14.745; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-1e8c9727083a.json)

### rg5-bcap · history-rg5-bcap-4ea2155c1f2d

**Scope:** historical; scientific; revision `219918317d615b8145dd427a4bb1537ff2a796dc27d364b9ecc56f1c71ae8f96`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 63.612; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-4ea2155c1f2d.json)

### rg5-bcap · history-rg5-bcap-550f292579bc

**Scope:** historical; scientific; revision `9a297426fb3c35fd7d4b2475b56dea5f5063ce1fa7ae365d2aa8aae577a90d5e`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 52.705; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-550f292579bc.json)

### rg5-bcap · history-rg5-bcap-55cd94940b92

**Scope:** historical; scientific; revision `b99b76f63bc6049c443d4f883e3302f792ac8bb19c7add0fc495f9b8aec2cbe1`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 18.525; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-55cd94940b92.json)

### rg5-bcap · history-rg5-bcap-576349693fa6

**Scope:** historical; scientific; revision `679ab9644316e2c83d6a019e861f95e6acc1196836d52447988015da6b03ece3`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 5.505; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-576349693fa6.json)

### rg5-bcap · history-rg5-bcap-5965d3f7a412

**Scope:** historical; scientific; revision `11830aaf946fa10d631d0b0a4702de9c8f564ab2f47e5cb3933087796c7c30cb`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 10.676; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-5965d3f7a412.json)

### rg5-bcap · history-rg5-bcap-5beabbff595a

**Scope:** historical; scientific; revision `826c2968ab9c6a29da5b07597a4ec98e2f4345e88cd7b7606d070d806dffdcf4`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 70.727; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-5beabbff595a.json)

### rg5-bcap · history-rg5-bcap-660b4979bba6

**Scope:** historical; scientific; revision `4d16badceeca3c0286ea460f364c871a45f5bb361e30c1368da944e2f9c3baf3`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1, 'NOT_RUN': 1}; wall seconds 344.993; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1, NOT_RUN=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-660b4979bba6.json)

### rg5-bcap · history-rg5-bcap-6ded4987ff55

**Scope:** historical; scientific; revision `522c22d041613c7e9ddebc3819ac07220232004df82da986c8aa3d9b5ee58098`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 59.705; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-6ded4987ff55.json)

### rg5-bcap · history-rg5-bcap-8a3d2c8b9a0c

**Scope:** historical; scientific; revision `b698c6b06ff09ea4933164b54068247ac18a6a145377eacc77b9fcfb9b8e6de6`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 4, 'PASS': 18}; wall seconds unknown; mechanism `training_formulation`.

ae_gan_hold: recon_mse=0.0061314; img_bars4: modes=4, hq=1

Historical recorded task outcomes: FAIL=4, PASS=18

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/qualification-summary.json) · [Record](records/history-rg5-bcap-8a3d2c8b9a0c.json)

### rg5-bcap · history-rg5-bcap-9c7ee6955939

**Scope:** historical; scientific; revision `b87a4fe888eb88abf2033865b97bc7cd149225d75e048f3888c7ff95b0ffd8c5`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 79.687; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-9c7ee6955939.json)

### rg5-bcap · history-rg5-bcap-9fd65df51556

**Scope:** historical; scientific; revision `aee966c263b10546da4c82150567631ac159f19dc7429ece4ed9c8e9524fe3ca`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 158.216; mechanism `unknown`.

rotated100: modes=100, hq=0.98585, precision=0.98585, mass_tv=0.04325

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-9fd65df51556.json)

### rg5-bcap · history-rg5-bcap-a56da07ada71

**Scope:** historical; scientific; revision `b7bceb7e5557ac72089853430d49cbfcc094cd7c8234593c66f03a47b9b27ea4`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 27.757; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-a56da07ada71.json)

### rg5-bcap · history-rg5-bcap-abe92485383c

**Scope:** historical; scientific; revision `8ec556bd069b5f61b08115676dd514e7025d5cdb5bc4eb5534b2492547a9d739`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 209.247; mechanism `unknown`.

mode_hold:shift: modes=6, hq=0.82324

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-abe92485383c.json)

### rg5-bcap · history-rg5-bcap-b6bfd45b1554

**Scope:** historical; scientific; revision `e175abfdb1d91588df75879f9a4dffcfbf159fde5382838ba979b6cc68298b78`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 63.327; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-b6bfd45b1554.json)

### rg5-bcap · history-rg5-bcap-b6e145fc72c1

**Scope:** historical; scientific; revision `92dac014115eeb11585c5269d5cff05f025b323cd00882d770766a1424dc360e`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 57.937; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-b6e145fc72c1.json)

### rg5-bcap · history-rg5-bcap-b99cc61e48d5

**Scope:** historical; scientific; revision `22f2818502b2a775a46d4f16116e4f3affdc1f77310cdecc4ca97ee685ce3692`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 16.906; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-b99cc61e48d5.json)

### rg5-bcap · history-rg5-bcap-be80728d94cb

**Scope:** historical; scientific; revision `922f1333b14b374cdff41e85b078e046a0f64831adba53045552eef23e76d78a`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 33.316; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-be80728d94cb.json)

### rg5-bcap · history-rg5-bcap-c273d601a176

**Scope:** historical; scientific; revision `ba4068f2b71d42a968c0d0ad3a5fcc9316a5b995bca9775e124063341e9cbd4d`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 56.364; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-c273d601a176.json)

### rg5-bcap · history-rg5-bcap-f23ed4abc1ff

**Scope:** historical; scientific; revision `8252629894bb63cac7f312f8e22453d0793d9ce2d00a8ed505bcdadaa33e807c`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 27.052; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-f23ed4abc1ff.json)

### rg5-bcap · history-rg5-bcap-f606f6c7f929

**Scope:** historical; scientific; revision `c41c87208009cd75de30d84eadf2e70c8bac7c8620c4d3e90ff6656a1fc6a0ed`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 29.691; mechanism `unknown`.



Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-f606f6c7f929.json)

### rg5-bcap · history-rg5-bcap-fd2a95d5c70c

**Scope:** historical; scientific; revision `4c32e861689880cba7ec62d13fc9d884113e32391466522cd60a3da683c05b00`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 45.511; mechanism `unknown`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/92dc03195080376ba6b56d273e5c3b929e54ce61/reports/toy100/gap-fill-20260925/results-summary.json) · [Record](records/history-rg5-bcap-fd2a95d5c70c.json)

### row-em-renew · history-row-em-renew-7f6a6784a6af

**Scope:** historical; scientific; revision `a126636b5950984940bace3df8740ba0a243e1704627789503179ccc2ca15c76`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 1691.55; mechanism `sampling_only_patch`.

grid100: modes=100, hq=0.9816, precision=0.9816, acc_center_rms_sigma=0.13017, mass_tv=0.0406, acc_radial_ks=0.01307

Historical recorded task outcomes: PASS=1 Public sampling calibration changes the sampling law; this is not a training-mechanism improvement.

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/grid100-result.json) · [Record](records/history-row-em-renew-7f6a6784a6af.json)

### row-em-renew · history-row-em-renew-9db09d7aab75

**Scope:** historical; scientific; revision `3694457d9e7cc9adbcb894d27fbabba58e9cfc42944684c0894b04ba5685ea28`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 1538.92; mechanism `sampling_only_patch`.

rotated100: modes=100, hq=0.9832, precision=0.9832, acc_center_rms_sigma=0.15228, mass_tv=0.03655, acc_radial_ks=0.028056

Historical recorded task outcomes: PASS=1 Public sampling calibration changes the sampling law; this is not a training-mechanism improvement.

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/rotated100-result.json) · [Record](records/history-row-em-renew-9db09d7aab75.json)

### row-em-renew · history-row-em-renew-ae1f17f6f4e5

**Scope:** historical; scientific; revision `908c27776642f7c184b25aa1008ee450dda4b9413af9d7ce3b86264e1a789d7d`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 1362.67; mechanism `sampling_only_patch`.

staggered100: modes=100, hq=0.98795, precision=0.98795, acc_center_rms_sigma=0.12412, mass_tv=0.04035, acc_radial_ks=0.009198

Historical recorded task outcomes: PASS=1 Public sampling calibration changes the sampling law; this is not a training-mechanism improvement.

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/staggered100-result.json) · [Record](records/history-row-em-renew-ae1f17f6f4e5.json)

### row-em-renew-all22 · history-row-em-renew-all22-c9a100c54f2f

**Scope:** historical; scientific; revision `17b592f92b62a61b22a4ab9845915971321649be421a09df5ea13e9c74b07a54`.

Unknown; consult the linked source narrative.

**Observed:** {'BLOCKED': 8, 'FAIL': 4, 'PASS': 10}; wall seconds unknown; mechanism `sampling_only_patch`.



Historical recorded task outcomes: BLOCKED=8, FAIL=4, PASS=10 Public sampling calibration changes the sampling law; this is not a training-mechanism improvement.

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/all22-summary.json) · [Record](records/history-row-em-renew-all22-c9a100c54f2f.json)

### row-em-sampling · history-row-em-sampling-86946821eed9

**Scope:** historical; scientific; revision `203b2133384ec9315d32a2b0cad4bc59e4392f37df6e30a84da9ee68caac4075`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 349.96; mechanism `sampling_only_patch`.

grid100: modes=100, hq=0.9836, precision=0.9836, acc_center_rms_sigma=0.18424, mass_tv=0.0476, acc_radial_ks=0.020449

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-sampling/grid100-result.json) · [Record](records/history-row-em-sampling-86946821eed9.json)

### row-em-sampling · history-row-em-sampling-8e2ea78eaf31

**Scope:** historical; scientific; revision `9f5b2622a2c5bbbde68079557b88e1d6c85091bc1fc0f382881e9f3bc6f6a228`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 349.71; mechanism `sampling_only_patch`.

rotated100: modes=100, hq=0.97835, precision=0.97835, acc_center_rms_sigma=0.19636, mass_tv=0.04445, acc_radial_ks=0.02439

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-sampling/rotated100-result.json) · [Record](records/history-row-em-sampling-8e2ea78eaf31.json)

### row-em-sampling · history-row-em-sampling-b08f0852490a

**Scope:** historical; scientific; revision `88e4765613ab6b43d833286ac4d3c8b1cc430aa72fd44bc0c0474a49210a6324`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 284.6; mechanism `sampling_only_patch`.

staggered100: modes=100, hq=0.98775, precision=0.98775, acc_center_rms_sigma=0.15653, mass_tv=0.0387, acc_radial_ks=0.0059515

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-sampling/staggered100-result.json) · [Record](records/history-row-em-sampling-b08f0852490a.json)

### rs1-dtrack · history-rs1-dtrack-12480461dcb8

**Scope:** historical; scientific; revision `07a5e697a836fbfe3bcc4416749f6f35461753cecd4624c48130cd3fabca4094`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 340.63; mechanism `parameter_change`.

staggered100: modes=100, hq=0.9713, precision=0.9713, acc_center_rms_sigma=0.19991, mass_tv=0.03575, acc_radial_ks=0.012897

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/dtrack2/rs1-dtrack-staggered100-result.json) · [Record](records/history-rs1-dtrack-12480461dcb8.json)

### rs1-dtrack · history-rs1-dtrack-1f6d6964791e

**Scope:** historical; scientific; revision `02150ba7b90d3fd5fe56a0eb15ec7ca7e017b7b09f9d14853fecde11223de810`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 455.02; mechanism `parameter_change`.

rotated100: modes=100, hq=0.9571, precision=0.9571, acc_center_rms_sigma=0.19295, mass_tv=0.039, acc_radial_ks=0.0055068

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/dtrack2/rs1-dtrack-rotated100-result.json) · [Record](records/history-rs1-dtrack-1f6d6964791e.json)

### rs1-dtrack · history-rs1-dtrack-e63fb8762624

**Scope:** historical; scientific; revision `381c6408dd97d10f019845c4036ec0a6a639a347b218898aa6c8ee1704343ce9`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 459.55; mechanism `parameter_change`.

grid100: modes=100, hq=0.9831, precision=0.9831, acc_center_rms_sigma=0.18834, mass_tv=0.03005, acc_radial_ks=0.018761

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/dtrack2/rs1-dtrack-grid100-result.json) · [Record](records/history-rs1-dtrack-e63fb8762624.json)

### rs1b-dtrack15 · history-rs1b-dtrack15-9fd3672dd965

**Scope:** historical; scientific; revision `08cbaee92bec17a81c4ff44942e1ba9fd1acb461c0d83f6b90b6f02dfc92df28`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 469.72; mechanism `parameter_change`.

rotated100: modes=100, hq=0.9525, precision=0.9525, acc_center_rms_sigma=0.23611, mass_tv=0.0337, acc_radial_ks=0.044693

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/dtrack2/rs1b-dtrack15-rotated100-result.json) · [Record](records/history-rs1b-dtrack15-9fd3672dd965.json)

### st-10-c22 · history-st-10-c22-6815082d4e4a

**Scope:** historical; scientific; revision `e9068ddba0056793a11f663ccbcfc582ba8042c0c8ad7edd4e633f78d73e1901`.

st-10 (stationarity LR + learnable sigma) on the 8 custom hosts (custom22 port)

**Observed:** {'FAIL': 3, 'PASS': 5}; wall seconds unknown; mechanism `unknown`.

ae_gan_hold: recon_mse=0.010027

Historical recorded task outcomes: FAIL=3, PASS=5

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/custom22/results.json) · [Record](records/history-st-10-c22-6815082d4e4a.json)

### st10-directadam · history-st10-directadam-0b1367956e4c

**Scope:** historical; scientific; revision `95fe7a2cf011f873c6d8f4bdabf1b2f83d8b079731a6dcc0b18b32593c2e940d`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 44.96; mechanism `training_formulation`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/two_pole.json) · [Record](records/history-st10-directadam-0b1367956e4c.json)

### st10-directadam · history-st10-directadam-62fc307f9f56

**Scope:** historical; scientific; revision `7b1106e876017b9d383b0923ef7d25c431859590182e757afc7129b0803f0ba0`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 4.71; mechanism `training_formulation`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/unipolar.json) · [Record](records/history-st10-directadam-62fc307f9f56.json)

### st10-directadam · history-st10-directadam-7788ef45e5a2

**Scope:** historical; scientific; revision `93b3a55dd59ff2300ff408ae42e9c844b5aec98dbcb20ae1da6b36f1112db8aa`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 8.19; mechanism `training_formulation`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/cover_leftover.json) · [Record](records/history-st10-directadam-7788ef45e5a2.json)

### st10-directadam · history-st10-directadam-79eaf7243172

**Scope:** historical; scientific; revision `39a3fe27b0e1b151a61344b4a0ce06640113e179f074bb22a4a3f0a8485d7be2`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 4.81; mechanism `training_formulation`.



Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/trajectory.json) · [Record](records/history-st10-directadam-79eaf7243172.json)

### st10-directadam · history-st10-directadam-7e84cfdd3f2f

**Scope:** historical; scientific; revision `ce5f8f3494593177ea986b45ca3ed0becfe40988b57e0bce3acefec6f7abc6e1`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 4.85; mechanism `training_formulation`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/residual_student.json) · [Record](records/history-st10-directadam-7e84cfdd3f2f.json)

### st10-directadam · history-st10-directadam-9f795f98c6fc

**Scope:** historical; scientific; revision `a75b22540ae8cedbc2d3a3e1b8034d58bb8b118dbeef662291af81507e958f3e`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 3.23; mechanism `training_formulation`.



Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/unused_token_hold.json) · [Record](records/history-st10-directadam-9f795f98c6fc.json)

### st10-directadam · history-st10-directadam-a9502996f0c8

**Scope:** historical; scientific; revision `a8237af278c8d6d9f7224e2812fc7e2844fb3a733fcdb446c6cff9cc4c303925`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 8.7; mechanism `training_formulation`.



Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/mid_scale_identity.json) · [Record](records/history-st10-directadam-a9502996f0c8.json)

### st10-directadam · history-st10-directadam-d9d4d5cc4807

**Scope:** historical; scientific; revision `6056d2588608c27140a51e3eb1533ec43e3b9224b54d089fa087a217676c5da9`.

Unknown; consult the linked source narrative.

**Observed:** {'PASS': 1}; wall seconds 4.51; mechanism `training_formulation`.

ae_gan_hold: recon_mse=0.010027

Historical recorded task outcomes: PASS=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/results/ae_gan_hold.json) · [Record](records/history-st10-directadam-d9d4d5cc4807.json)

### st5-dbase-qr · history-st5-dbase-qr-09a7c5a332fb

**Scope:** historical; scientific; revision `a8ee186c328268319514d888be931aa6731ec69fa077ffdfd135931ffa2fae6a`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds unknown; mechanism `training_formulation`.

grid100: modes=99, hq=0.9821, precision=0.9821, acc_center_rms_sigma=0.31529, mass_tv=0.0598, acc_radial_ks=0.087259

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/native100-results.json) · [Record](records/history-st5-dbase-qr-09a7c5a332fb.json)

### st5-scaleaware-qr-14k · history-st5-scaleaware-qr-14k-4cb010ecea61

**Scope:** historical; scientific; revision `62d57499f45786f6c9e96fea59cd97b9692d6989076d33794f30aac0cc6d8b45`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds unknown; mechanism `training_formulation`.

rotated100: modes=100, hq=0.97475, precision=0.97475, acc_center_rms_sigma=0.30753, mass_tv=0.0314, acc_radial_ks=0.068666

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/native100-results.json) · [Record](records/history-st5-scaleaware-qr-14k-4cb010ecea61.json)

### st5-scaleaware-qr · history-st5-scaleaware-qr-1dc7735192d9

**Scope:** historical; scientific; revision `a0ea80a712727da9ab179b59e8ca2d822c2a9d9208a99554dbd56d8de9ed6d70`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 3}; wall seconds unknown; mechanism `training_formulation`.

grid100: modes=100, hq=0.97865, precision=0.97865, acc_center_rms_sigma=0.3638, mass_tv=0.05885, acc_radial_ks=0.084431; rotated100: modes=100, hq=0.96185, precision=0.96185, acc_center_rms_sigma=0.28144, mass_tv=0.03425, acc_radial_ks=0.11995; staggered100: modes=100, hq=0.95955, precision=0.95955, acc_center_rms_sigma=0.52172, mass_tv=0.048, acc_radial_ks=0.067346

Historical recorded task outcomes: FAIL=3

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/native100-results.json) · [Record](records/history-st5-scaleaware-qr-1dc7735192d9.json)

### st6-eg-qr · history-st6-eg-qr-f117ca0bd29b

**Scope:** historical; scientific; revision `e2fd4acbce3eeafcab71078e29fe117ce197e5ecc3bf4a4a21f35e9e9a0f9a60`.

Unknown; consult the linked source narrative.

**Observed:** {'INCOMPLETE': 1, 'NOT_RUN': 2}; wall seconds unknown; mechanism `training_formulation`.



Historical recorded task outcomes: INCOMPLETE=1, NOT_RUN=2

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/native100-results.json) · [Record](records/history-st6-eg-qr-f117ca0bd29b.json)

### st7-fake-reopen-qr · history-st7-fake-reopen-qr-e0d3509f57c7

**Scope:** historical; scientific; revision `ca3b0a25790e3d7c0fba11a12fd97213bdf76e058cc607f2ce1a6aa24ac58ec6`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds unknown; mechanism `training_formulation`.

grid100: modes=100, hq=0.97865, precision=0.97865, acc_center_rms_sigma=0.3638, mass_tv=0.05885, acc_radial_ks=0.084431

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/native100-results.json) · [Record](records/history-st7-fake-reopen-qr-e0d3509f57c7.json)

### unresolved:reports/toy100/lrfree-search/direct-particle-adam/summary.json · history-unresolved-reports-toy100-lrfree-search-direct-particle-adam-sum-5d6d73a3abe4

**Scope:** historical; scientific; revision `ed0cb655262ea352b82b227115317fa29fa40ac6e6a7acc2cbec70a3aa03d586`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 2, 'PASS': 6}; wall seconds unknown; mechanism `training_formulation`.

two_pole: passing_checks=7, observations=24; trajectory: passing_checks=0, observations=24; residual_student: passing_checks=21, observations=24

Historical recorded task outcomes: FAIL=2, PASS=6

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/direct-particle-adam/summary.json) · [Record](records/history-unresolved-reports-toy100-lrfree-search-direct-particle-adam-sum-5d6d73a3abe4.json)

### v3_pair_graft · history-v3-pair-graft-126441aebe6e

**Scope:** historical; scientific; revision `6322709fde85358c9ead44d7270f9c647f23a944162c44e5b78f614d19438882`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 352.92; mechanism `unknown`.

grid100: modes=100, hq=0.97365, precision=0.97365, acc_center_rms_sigma=0.2595, mass_tv=0.0388, acc_radial_ks=0.01173

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/paired-bd-graft/graft-result.json) · [Record](records/history-v3-pair-graft-126441aebe6e.json)

### v3_repro · history-v3-repro-50fe3393d8b8

**Scope:** historical; scientific; revision `ec24b50ffdf9ab758c73b50983b40986d2cc3b590ed66742fc405264aa071c47`.

Unknown; consult the linked source narrative.

**Observed:** {'FAIL': 1}; wall seconds 224.86; mechanism `unknown`.

grid100: modes=100, hq=0.97325, precision=0.97325, acc_center_rms_sigma=0.30273, mass_tv=0.06095, acc_radial_ks=0.055281

Historical recorded task outcomes: FAIL=1

**Next:** Bind compatible task, fixture, package, scoring, and complete raw evidence before qualification reuse.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/paired-bd-graft/baseline-result.json) · [Record](records/history-v3-repro-50fe3393d8b8.json)

### k3p-no-output-noise-diagnostic · lifecycle-1fdc4834581ce2e17a87e8e7

**Scope:** administrative; administrative; revision `0fa6918ca4d0456515de09dbcfaeedffba1d4cecaab00dee5028bcf3b0b34603`.

Removing additive training output noise may prevent the clean learned-MoG native distribution from contracting late. This is a one-factor diagnostic of the training/scoring-law mismatch, not a predicted positive or a full-reference claim.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `structural`.



Administrative abandoned: Concluded preregistered full-native FAIL. Training output-noise removal improves shape but fails sustained coverage, all five terminal accuracy checks and independent holdout. Stop this revision; no further smoke/reference/14k spending. New profile cannot meet positive-reference plus zero false-rejection criteria.

**Next:** Retain this revision's evidence and readout; declare a new revision before further work.

[Evidence](../../) · [Record](records/lifecycle-1fdc4834581ce2e17a87e8e7.json)

### atlas-particle-normal-v4 · lifecycle-6c54ed3e24538152abaee6dc

**Scope:** administrative; administrative; revision `79e308b48506dccbf888516575643268a7499751b98b7d9802fbf7c56b2ad186`.

None

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `structural`.



Administrative superseded: Pre-fix partial run on commit 6c895b5924d0 superseded by clean reverified v4 execution at efebaebc63fb on commit 6261a5458c1f.

**Next:** See successor atlas-particle-normal-v4@efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd.

[Evidence](../../) · [Record](records/lifecycle-6c54ed3e24538152abaee6dc.json)

### k3p-bcap-matched-v1 · lifecycle-78a4ebc4a9d5f08ddeab2fa9

**Scope:** administrative; administrative; revision `5db772f81cafe59b642b5707e600380fb49c3d42623b5c769e03dd677bbbfafd`.

Replace only the critic penalty by frozen historical bcap while retaining the K3P optimizer, noise, A2, host and sampling contracts.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `structural`.



Administrative abandoned: Registered native comparison failed the unchanged full gates; stop this exact revision without further matrix filling, tuning or continuation. Preserve its concluded readout and complete diagnostic evidence.

**Next:** Retain this revision's evidence and readout; declare a new revision before further work.

[Evidence](../../) · [Record](records/lifecycle-78a4ebc4a9d5f08ddeab2fa9.json)

### release07-gan-v3-cloud-v1 · lifecycle-a7f0052186f7f049b3b6cf54

**Scope:** administrative; administrative; revision `3f2629ea8a1a6c9176d021ccdb99df48c2f7f67296946110288d59f3015ae14d`.

Evaluate the released v0.7 GAN v3 effective recipe on its explicit z4/batch256/unstandardized cloud host with named caller initialization.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `structural`.



Administrative abandoned: Registered native comparison failed the unchanged full gates; stop this exact revision without further matrix filling, tuning or continuation. Preserve its concluded readout and complete diagnostic evidence.

**Next:** Retain this revision's evidence and readout; declare a new revision before further work.

[Evidence](../../) · [Record](records/lifecycle-a7f0052186f7f049b3b6cf54.json)

### k3p-r1r2-matched-v1 · lifecycle-afc11349bf0ada8bca0c4c89

**Scope:** administrative; administrative; revision `3bff4e5835a30fd018893f3b3cb1b612c5ba28d39e62247b61b728b38987b85e`.

Replace only the critic penalty by frozen historical r1r2 while retaining the K3P optimizer, noise, A2, host and sampling contracts.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `structural`.



Administrative abandoned: Registered native comparison failed the unchanged full gates; stop this exact revision without further matrix filling, tuning or continuation. Preserve its concluded readout and complete diagnostic evidence.

**Next:** Retain this revision's evidence and readout; declare a new revision before further work.

[Evidence](../../) · [Record](records/lifecycle-afc11349bf0ada8bca0c4c89.json)

### release07-gan-v3-mog-v1 · lifecycle-e90204eef53f90292d2d1750

**Scope:** administrative; administrative; revision `a9b2aa0ace96762b024a30c0979a0eedce8d07f9fae5cb5dc03868522132962c`.

Evaluate the full v0.7 GAN v3 dynamics adapted only to the matched affine z2/batch2048/MoG host.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `structural`.



Administrative abandoned: Registered native comparison failed the unchanged full gates; stop this exact revision without further matrix filling, tuning or continuation. Preserve its concluded readout and complete diagnostic evidence.

**Next:** Retain this revision's evidence and readout; declare a new revision before further work.

[Evidence](../../) · [Record](records/lifecycle-e90204eef53f90292d2d1750.json)

### bcap-pure-budget10x-v1 · pr311-bcap-pure-budget10x-context-v1

**Scope:** historical; family_context; revision `beed204f8ee2cf15d4b8f1ddbc5ff41ffe5fcd30`.

Does 10x longer training with the same constant-step BCAP-pure recipe fix the Gaussian and Ring16 sustained Tier 1 failures?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `training_budget_extension`.



No: 0/2 sustained passes. Gaussian keeps oscillating and fails distribution accuracy; Ring16 removes most tail error but repeatedly loses component spread and fails the five-terminal-observation hold. Closed unmerged in PR triage 2026-10-10; superseded by develop's 22/22 Tier 1 pass.

**Next:** Do not use longer budgets as a fix for BCAP oscillation or ring hold; change the dynamics instead.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/beed204f8ee2cf15d4b8f1ddbc5ff41ffe5fcd30/reports/forge/bcap-pure-budget10x-v1/README.md) · [Record](records/pr311-bcap-pure-budget10x-context-v1.json)

### bcap-gaussian-frozen-prior · pr318-gaussian-frozen-prior-context-v1

**Scope:** historical; family_context; revision `dcbcd3ed72b0b3d19bc2f4d52443574dc44d3d4e`.

Does freezing the initial MoG prior let G/D acquire and retain the 1D Gaussian at the selected constant BCAP rates?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `fixed_prior_control`.



No: a frozen prior does not rescue Gaussian retention under alternating, simultaneous or past-extrapolation timing; extrapolation reaches a passing final shifted snapshot but cannot retain full quality, and all three fixed-prior ring arms fail. Explicit fixed-prior control cohort. Closed unmerged in PR triage 2026-10-10.

**Next:** Prior motion is not the cause of Gaussian retention failure; look at G/D dynamics.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/dcbcd3ed72b0b3d19bc2f4d52443574dc44d3d4e/reports/forge/gaussian-frozen-prior/README.md) · [Record](records/pr318-gaussian-frozen-prior-context-v1.json)

### ring16-tiny-weak-subspace-noise · pr335-ring16-tiny-noise-context-v1

**Scope:** historical; family_context; revision `a570d722949b4bf26f3641b8d020a7a1cf67ec3b`.

Does continuous tiny noise on weak gradient subspaces fix Ring16 acquisition and hold?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `weak_subspace_gradient_noise`.



Yes: every-update tiny weak-subspace noise passes confirmed Ring16 smoke (acquires at 817) and ends with 45 consecutive full passes; noise at update 401 only fails confirmation on covariance. Slower than merged rank truncation (acquires 684, develop default via PR332). Closed unmerged in PR triage 2026-10-10; absorbed into PR331 Ring16 report.

**Next:** Fallback candidate if truncation is removed or regresses; not needed while truncation is default.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/a570d722949b4bf26f3641b8d020a7a1cf67ec3b/reports/forge/ring16-noise/README.md) · [Record](records/pr335-ring16-tiny-noise-context-v1.json)

### bcap-runtime-margin-pr342-pr343 · pr342-343-bcap-gate-margin-context-v1

**Scope:** historical; family_context; revision `ba593aab5c0e150da1c36005aece17816f074c7b`.

Did serial autograd scheduling and/or numerical rank truncation cause the BCAP Gaussian-smoke and five-word-hold regressions?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `runtime_scheduling_and_rank_truncation_sensitivity`.



BCAP gate margins are thin. On Gaussian smoke (PR342), disabling autograd multithreading (serial scheduling) flips PASS to FAIL for both polar rules (KS .0692/.0646 vs limit .05); truncation alone does not. On five-word hold (PR343), serial scheduling alone and rank truncation alone each flip the historical PASS to FAIL; together they reproduce V6 FAIL. All word arms still acquire; the regression is retention. Closed unmerged in PR triage 2026-10-10; superseded by develop's smoothing=1e-5 + direction_blend 22/22 Tier 1 pass.

**Next:** Treat runtime scheduling and polar truncation as confounders when attributing BCAP gate changes; compare under matched runtime before crediting a formulation change.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/ba593aab5c0e150da1c36005aece17816f074c7b/reports/forge/bcap-gaussian-regression/README.md) · [Record](records/pr342-343-bcap-gate-margin-context-v1.json)

### bcap-smooth-polar-1e-4 · pr344-smoothed-dualnorm-1e-4-context-v1

**Scope:** historical; family_context; revision `6d3d5b21144d85061fce6349b706e9c2800b0f96`.

Does a fixed smoothed DualNorm scale of 1e-4 repair both serial Gaussian acquisition and word stability across the four truncation/threading combinations?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `smoothed_polar_dualnorm_scale`.



Smoothing 1e-4 repairs serial Gaussian acquisition and Ring16 passes in all four combinations, but words fail the original terminal stability gate in all four runs. 1e-5 was selected instead (merged PR345). Closed unmerged in PR triage 2026-10-10.

**Next:** Keep smoothing at the selected 1e-5; do not raise it to 1e-4 without a word-stability falsifier.

[Evidence](https://github.com/255BITS/ParticleGAN/blob/6d3d5b21144d85061fce6349b706e9c2800b0f96/reports/forge/smooth-polar-factorial/README.md) · [Record](records/pr344-smoothed-dualnorm-1e-4-context-v1.json)

### r1r2-saved-endpoint-diagnosis · r1r2-failure-diagnosis-20261022

**Scope:** saved_state_diagnostic; family_context; revision `a771841dd17bd8780c03f8ebb6e34b8dfe19bc94`.

Saved public-API states and an independent software oracle distinguish R1/R2 implementation defects from density, schedule and categorical-retention failures without retraining.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `frozen_state_force_and_implementation_audit`.



No implementation defect or validated global recipe fix found. Original Ring16 failure is density quality despite all 16 modes. The strongest NS control has substantial critic gradients but its projected radial force contracts already narrow spread at both integration orders. Fixed-width MMD improves an output-space covariance derivative but worsens it after saved generator/prior and AMSGrad projection on both finite panels; it remains untrained. Selected word hold has 4/5 severely attenuated categorical atom gradients in float32 and float64 and no explicit reconstruction training loss. Independent G/D/learned-prior update oracle and adjacent suites: 110 tests pass. Original outcomes and the single current leaderboard remain unchanged.

**Next:** Keep MMD deferred and stop the concluded optimizer/energy/prior-freeze revisions. Treat 6400-update schedule PASS as a distinct fragile diagnostic, not a 1600-update repair. A word logit-space reconstruction intervention is only a future explicit recipe hypothesis. Require a supported generator/optimizer-aware mechanism and a fresh bounded matched campaign before any global recipe replacement.

[Evidence](../../reports/forge/r1r2-failure-diagnosis-20261022/README.md) · [Record](records/r1r2-failure-diagnosis-20261022.json)

### r1r2-pr401-concluded-archive · r1r2-pr401-concluded-archive-v1

**Scope:** historical; family_context; revision `2cf7b0a4c796e29ace6cca12d2e553ad67211b5c`.

Historical R1/R2 global recipe repairs preserve the original task/protocol gates; this reporting archive proposes no new experiment.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `concluded_research_archive`.



All 34 complete measured revisions fail Ring16. Strongest complete repairs retain 5/6 ordinary tasks plus clock PASS; original scheduled baseline has 5/6 plus clock FAIL. Energy-U and frozen-prior energy both fail covariance/HQ on all 96 saved Ring16 panels. All original acquisition/endpoint distinctions and sources remain unchanged; MMD is unmeasured deferred preparation.

**Next:** Retain the stronger measured control and negative findings; stop these exact repairs. Accept compact evidence-only replacement before closing original mixed PR #401. Defer MMD runtime and paid execution; no budget change, qualification reuse or rerun.

[Evidence](../../reports/forge/r1r2-concluded-archive/README.md) · [Record](records/r1r2-pr401-concluded-archive-v1.json)

### bispf-optimized-quintic-v1 · readout-01bff4a5f0a8f6d91bc51c65

**Scope:** research_diagnostic; scientific; revision `910c746cc18347ce72d108dcef81f283fc5f2efd2117fc6358f7b485ba172ae1`.

A conditional energy score added to BISPF improves low-noise conditional-law calibration; the matched GPU0 selected-recipe control may retain the historical Gaussian failure.

**Observed:** {'FAIL': 1, 'INCOMPLETE': 1}; wall seconds 1019.358; mechanism `structural`.



MatchedGPU0 selected-recipe Gaussian completes2500updates and FAILS; bimodal is resource INCOMPLETE at600s.

**Next:** Reuse complete Gaussian baseline exactly; finish separately registered1200s bimodal resource variant alongside the bounded global energy mobility repair. No unchanged Gaussian rerun or ordinary qualification.

[Record and receipt provenance](records/readout-01bff4a5f0a8f6d91bc51c65.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-r1r2-matched-v1 · readout-0297c36f9f2d9c5feb45b3cb

**Scope:** calibration_diagnostic; scientific; revision `3bff4e5835a30fd018893f3b3cb1b612c5ba28d39e62247b61b728b38987b85e`.

Replace only the critic penalty by frozen historical r1r2 while retaining the K3P optimizer, noise, A2, host and sampling contracts.

**Observed:** {'FAIL': 1}; wall seconds 83.983; mechanism `structural`.

grid100_affine_paired_laws_v1: precision=0.97753, center_rms_sigma=0.3375, mass_tv=0.02802, radial_ks=0.059452

Preregistered full7000-update grid100_affine_paired_laws_v1 FAIL; sustained coverage FAIL, 0/5 terminal joint passes. Independent100k clean/live holdout: centre RMS 0.337500287sigma, covariance bias -0.186036085, radialKS 0.059452130, massTV 0.028020000, precision 0.97753. All G/D/prior updates complete, finite/hooks checks PASS, unintended RNG deviations0, independent receipt/artifact/evaluator audit PASS. Unique paid 83.982570446999s. No ordinary qualification or promotion; all unknown cells preserved.

**Next:** Stop these exact revisions and this selected registration. Before another GPU study, perform a bounded zero-training saved-state audit of critic radial and centering gradients against per-mode moments. Only a supported substantive hypothesis warrants a new frozen registration. Preserve A2 and sigma .025; no coefficient, cap, width, amplitude or seed sweep, best-checkpoint/EMA substitution, 14k continuation or automatic matrix expansion.

[Record and receipt provenance](records/readout-0297c36f9f2d9c5feb45b3cb.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-prior-fresh-gaussian-v1 · readout-030ba1dfffb896ffe7ef8768

**Scope:** research_diagnostic; scientific; revision `373a307163ee384dbed023239eda5b56bdce641eef92a5e8a5f87c8d3b54019c`.

Replacing table codes with fresh N(0, I) latents of the same dimension loses sampled-latent passes that the learned BISPF default holds.

**Observed:** {'INCOMPLETE': 15}; wall seconds 51.643; mechanism `structural`.



Cancelled: every attempt crashed at prior construction (Recipe rejected fresh_gaussian during replace(prior_update=learned)); repaired and rerun as prior-ablation-a3-fresh-gaussian-r2. INCOMPLETE receipts preserved.

**Next:** Superseded by prior-ablation-a3-fresh-gaussian-r2.

[Record and receipt provenance](records/readout-030ba1dfffb896ffe7ef8768.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v2 · readout-05eaed46ab0e45214ae8e7b4

**Scope:** current; scientific; revision `fc09c6b1f8ae77167ca31db0a4d9ef252dc9d6b92aed98e3d4b53a168ff0ac15`.

Adding clock-free prototype-resolution-scaled settling Born-Infeld sharpness (ebi_beta=6.0 tightening as ema_residual settles below 0.16) and discrete-simplex post-acquisition Sinkhorn thermal cooling enables empf-v2 to retain 100% Tier 1 qualification (11/11 required + clockfree_audit_measurement_v1) with zero regressions over empf-v1 while passing both Tier 2 mode_hold (hq >= 0.90) and five_word_joint_hold under one global constant-LR DualNorm recipe.

**Observed:** {'FAIL': 16, 'INCOMPLETE': 1, 'PASS': 22}; wall seconds 4054.537; mechanism `structural`.



On ordinary discriminator_stability (Rev 16) under protocol seed 0 on NVIDIA RTX A6000, empf-v2 passed 11/11 required Tier 1 tasks (100% Tier 1 qualification: gaussian1d_smoke, two_pole, unused_token_hold, ae_gan_hold, ring16_acquisition, five_word_joint_smoke, zero_tanh_smoke, model_ae_gan_multimodal, model_ae_reconstruction, model_vae_categorical, model_vae_hard) plus clockfree_audit_measurement_v1 diagnostic PASS, with zero regressions over empf-v1. Advancing to Tier 2, empf-v2 observed its preregistered mode_hold prediction (final hq=0.9993 >= 0.90, 8/8 modes, passing_suffix=14) and passed five_word_joint_hold (25/25 scheduled joint generation and exact inverse checks, final mass_tv=0.0209, min_token_prob=0.9981), lifting EMPF to 10/18 required Tier 2 passes (22/38 executed tasks PASS overall; 85/115 counted view requirements).

**Next:** Publish empf-v2 as the selected current EMPF family measurement on discriminator_stability while preserving empf-v1 in historical_selections and keeping DEFAULT_RECIPE = bispf_optimized unchanged pending calibration and holdout confirmation; target prototype capacity K > B_r for 100-mode grids and per-step DDGAN conditional energy in future bounded revisions.

[Record and receipt provenance](records/readout-05eaed46ab0e45214ae8e7b4.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-no-critic-penalty · readout-060e7be73e2832e314b9573f

**Scope:** calibration_diagnostic; scientific; revision `1e1a3e447fc222392dd327f0ae3d17ecf4dd7e7ce26ff79f82d69c31668f4461`.

Removing the public critic gradient penalty tests whether cheap stability gates reject a substantively weakened formulation before quality work.

**Observed:** {'INCOMPLETE': 1, 'PASS': 1}; wall seconds 25.601; mechanism `structural`.

vector_two_broad: hq=0.97266, mass_tv=0.0075684

Learned-MoG vector_two_broad PASS after exactly one registered operational cancellation repair. This is a single task diagnostic, not full-reference or smoke qualification.

**Next:** Retain cancelled receipt/cost and final scientific pass. Do not repeat this scientific PASS or promote from one reference. Complete host-provenance review before choosing further bounded calibration cells.

[Record and receipt provenance](records/readout-060e7be73e2832e314b9573f.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v3 · readout-0c549ac1c635a42dc43cf6dc

**Scope:** current; scientific; revision `baae191aa8fc7e2d0563cd8a3b52e8ef53ece846d63213cadf159a2e2c47018e`.

Adding shock-filtered transport residual EMA, trimmed central-[0.04, 0.96] 1D quantile-W2 floor-subtracted 1/sqrt(B) thermal cooling with DualNorm step-scale damping, Poisson empty-cell prototype shift acceleration, and prototype-referenced deep harmonic-capture lock hysteresis enables empf-v3 to retain every Tier 1 and Tier 2 pass of empf-v2 while newly passing Tier 2 gaussian1d_stability (all 72 stationary checks, deadline reacquisition, and 24 shifted hold checks) as well as continuous-learning shift/endurance tasks (target_shift_recovery, ring_hold, ring_extension) under one global clock-free constant-LR DualNorm recipe.

**Observed:** {'FAIL': 2, 'INCOMPLETE': 1, 'PASS': 12}; wall seconds 1614.857; mechanism `structural`.

model_ae_reconstruction: modes=4, precision=0.26587, mass_tv=0.049805

On ordinary discriminator_stability (Rev 16) under protocol seed 0 on NVIDIA RTX A6000, empf-v3 stopped at Tier 1 (9/11 required Tier 1 PASS plus clockfree_audit_measurement_v1, precision_gaussian_fp32, and precision_gaussian_mxfp8 PASS; model_vae_categorical FAIL with passing_suffix=4 < 5 despite passing final metrics at step 1200; five_word_joint_smoke INCOMPLETE due to 900s wall-time timeout at step 15001/20001 with 17/18 completed checks passing). Tier 2 gaussian1d_stability was not reached on the official queue because Tier 1 required completion blocked promotion.

**Next:** Preserve empf-v3's Tier 1 readout receipt as negative/incomplete evidence, remove small-batch online prototype growth on 32 <= B_r < 100 and restore empf-v2's exact 2D+ Born-Infeld settling gain while reducing per-step discrete-simplex CPU-GPU synchronization overhead on five_word_joint_smoke in a new candidate empf-v4.

[Record and receipt provenance](records/readout-0c549ac1c635a42dc43cf6dc.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2-repair-control-20261022 · readout-0c965d99d82ddcae5d1d8d5a

**Scope:** current; scientific; revision `3cca6de5d931b16469b0ef0fedbc0b5cb21f55b8af202a7868e94277fcef2037`.

The exact archived stronger global R1/R2 reference retains its prior observed ordinary passes and still exposes the unresolved Ring16 density failure under one fresh matched source/runtime. This new-source operational repair fixes snapshot provider dependencies, preserves original runtime errors and routes the already-declared tanh evaluator; all original scientific gates and recipe values remain unchanged.

**Observed:** {'FAIL': 3, 'PASS': 11}; wall seconds 495.821; mechanism `structural`.

ae_gan_hold: recon_mse=0.0068211; five_word_joint_smoke: modes=5, mass_tv=0.018945

Complete unchanged current Tier1: 10/13 required PASS, 3 FAIL; separate clock PASS. Ring16, DDGAN Gaussian and AE-DDGAN Gaussian fail. The stronger NS/AMSGrad control reproduces its archived Ring endpoint and does not qualify globally.

**Next:** Retain R2 and the selected historical recipe; no ordinary Tier2 or default promotion. Finish separately scoped own-checkpoint word hold, publish all failures, and require a demonstrated density-force/conditional-tracking prerequisite before another global candidate.

[Evidence](../../reports/forge/r1r2-repair-20261022/README.md) · [Record](records/readout-0c965d99d82ddcae5d1d8d5a.json)

### recipe-prior-refactor-bcap-pure-v1 · readout-0df182292d888440f5fb32ce

**Scope:** current; scientific; revision `21fce6ca73c85eb35e4f58b73c17e05db60619759947cca48fbb1df01d9c6237`.

Ordinary re-qualification of the unchanged selected bcap-pure recipe (recipe-prior-refactor-bcap-pure-v1) on current develop source 2e8a97a0aa84 under protocol seed 0 on NVIDIA RTX A6000 records current-source discriminator_stability and shared Tier 1 goal-view verdicts.

**Observed:** {'INCOMPLETE': 1, 'PASS': 20}; wall seconds 2134.696; mechanism `structural`.

ae_gan_hold: recon_mse=0.004353; model_ae_gan_multimodal: modes=4, precision=0.99512, mass_tv=0.080078

Completed discriminator_stability seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (FAIL 7, PASS 7). Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact.

**Next:** Retain source-bound readout and compiled Forge reports.

[Evidence](../../) · [Record](records/readout-0df182292d888440f5fb32ce.json)

### forge-onboarding-anchor-ablation · readout-10242573330bec1494c7602b

**Scope:** current; scientific; revision `39dc1cc8e516f8127e732edea196f82243ff8cdff4ed761020f0f802a3bcaf0f`.

Removing the public critic EMA-anchor term may weaken short-horizon adversarial movement; the fixed CPU smoke gate should reject a harmful ablation before larger tests.

**Observed:** {'FAIL': 1}; wall seconds 5.975; mechanism `structural`.



The public EMA-anchor ablation failed two_pole after 80 fixed-seed CPU updates: mean_abs=0.1048249304 was below 0.30, while grad_med=0.07678461075 passed its <=1.0 bound. The independently graded terminal curve had 0 of 24 passing observations. One attempt consumed 5.974550719 seconds; no other smoke, quality or endurance task launched.

**Next:** Stop this ablation at tier 1. Keep its frozen FAIL receipt as onboarding and fail-fast evidence; do not promote it or spend on downstream tasks. A scientific anchor comparison would require one explicitly bounded compatible parent/ablation design after the initial gate calibration gaps are addressed.

[Record and receipt provenance](records/readout-10242573330bec1494c7602b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-10ed0ff3f18514315a760f4a

**Scope:** current; scientific; revision `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`.

The single global quintic convolution-rule delta preserves every incumbent BISPF pass on the unchanged ordinary task suite while reducing large-image update latency; direct wide-image diagnostics independently test the accelerated path.

**Observed:** {'FAIL': 11, 'PASS': 19}; wall seconds 3155.051; mechanism `structural`.

ae_gan_hold: recon_mse=0.0036687

The single global quintic convolution-rule delta preserves all 19 incumbent BISPF passes across the complete 30-task ordinary suite, with all 11 historical failures retained and no pass regressions. Both ring phases pass every 1200 hold and 300 extension check; actual 3282 updates satisfy the fixed first-convergence stop rule under the maximum 7500-update allowance. All jobs finish without cancellation or retry.

**Next:** Review the complete same-source performance matrix and direct wide-channel diagnostics together with this preserved full ordinary pass set before selecting one whole BISPF measurement row. Retain all failures, archived controls, exact source identities and unknown performance contention. No public-default adoption or natural-image quality claim follows automatically.

[Evidence](../../reports/forge/bispf-optimized/README.md) · [Record](records/readout-10ed0ff3f18514315a760f4a.json)

### bispf-optimized-quintic-v1 · readout-13bb0a182066511013f5e739

**Scope:** research_diagnostic; scientific; revision `62c29b5e265d2d5556010afa6e2f8c8387479ab4a3104ff2c72057b2577c19b8`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.74; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-13bb0a182066511013f5e739.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-143d85d4540548b0ecec1604

**Scope:** current; scientific; revision `6db8151c1a987602f94a8b9e854b003b7176fec3c3039cde07679d8960ae9c1b`.

Ordinary re-qualification of the unchanged selected bispf recipe (bispf-optimized-quintic-v1) on current develop source 2e8a97a0aa84 under protocol seed 0 on NVIDIA RTX A6000 records current-source discriminator_stability and shared Tier 1 goal-view verdicts.

**Observed:** {'INCOMPLETE': 1, 'PASS': 20}; wall seconds 2427.454; mechanism `structural`.

ring16_acquisition: modes=16, hq=0.9519, mass_tv=0.092041

Completed discriminator_stability seed-0 CUDA execution for bispf-optimized-quintic-v1 (FAIL 6, PASS 8). Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact.

**Next:** Retain source-bound readout and compiled Forge reports.

[Evidence](../../) · [Record](records/readout-143d85d4540548b0ecec1604.json)

### forge-no-critic-penalty · readout-15e86dd160a1d64a04ceee04

**Scope:** current; scientific; revision `8c2186b8233422534e7d854e115a44d0ed3b88bbce04097814c85f402df8f3b8`.

Removing the public critic gradient penalty tests whether cheap stability gates reject a substantively weakened formulation before quality work.

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 19.305; mechanism `structural`.

ae_gan_hold: recon_mse=0.004034

Corrected-source smoke matrix: two_pole FAIL, unused_token_hold FAIL, ae_gan_hold PASS. Whole-screen FAIL; independent reference outcomes remain unmeasured. All selected attempts completed, with actual named-stream seed labels now consistent.

**Next:** Keep ordinary qualification stopped and retain this frozen cohort. The two-cell learned-MoG GPU pilot is registered but must wait for the requested joint device window. Collect deeper independent references only through explicit diagnostic selections, then evaluate the unchanged calibration criteria before adoption.

[Record and receipt provenance](records/readout-15e86dd160a1d64a04ceee04.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2-repair-r1cycle-20261022 · readout-15fd8ba9f7b947395f31c418

**Scope:** research_diagnostic; scientific; revision `de3afa5bb4c97c9e51a2ef66817108ca2372525aebff8200537f9df9d3328421`.

This exact source-bound global recipe can retain its own earliest independently confirmed word acquisition through every original scheduled primary and independent confirmation check, including the exact restored state, for4000 further updates. The original word_hold evaluator and all25 paired checks are authoritative; endpoint token probability alone cannot establish retention. This diagnostic changes scheduler placement only and grants no ordinary qualification.

**Observed:** {'PASS': 2}; wall seconds 154.319; mechanism `structural`.

five_word_joint_hold: modes=5, mass_tv=0.020898; five_word_joint_smoke: modes=5, mass_tv=0.018945

Own-checkpoint acquisition and full4000-update hold PASS: all25 scheduled/restored-state checks confirmed, parent update834, endpoint minimum token probability0.9999691247940063. No ordinary qualification credit. The NS/AMSGrad reference already passes retention; explicit reconstruction improves confidence margin but is not necessary for these gates.

**Next:** Close the finite three-role campaign. Keep R2; do not promote the failed global repairs. Preserve all ordinary failures and archived source identities. Require a demonstrated density or conditional-field prerequisite before another candidate.

[Evidence](../../reports/forge/r1r2-repair-20261022/README.md) · [Record](records/readout-15fd8ba9f7b947395f31c418.json)

### k3p-a2-off-native-diagnostic · readout-170dc0a68d5b68737b370a5f

**Scope:** current; scientific; revision `b5ae40d1428d03591c42cb5cddb8adea5042caaa34c8ea2b131565c0ccca72e9`.

Diagnostic hypothesis: removing the active A2 sparse-row damping may change the late covariance undershoot on the unchanged named affine grid100 learned-MoG host. This isolates one mechanism; saved evidence does not predict a positive result, and no full-reference or adoption claim is made.

**Observed:** {'FAIL': 1}; wall seconds 5.426; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 0.0637243241071701, 'mean_abs': 0.10652091354131699}. One completed attempt cost 5.425771753 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/b86f62c3e4f84a43b85d366e4cc43824.json) · [Record](records/readout-170dc0a68d5b68737b370a5f.json)

### bispf-optimized-quintic-v1 · readout-1858b3294015f9149e7fcb9d

**Scope:** research_diagnostic; scientific; revision `a2bdf3d01a3c507b273d49bdefa1bf0707963aed72d1ed10fcca32bf82e4cf6d`.

Retain BISPF core and distinguish incomplete-data recovery from learned multimodal conditional amortization with the frozen scorer and controls.

**Observed:** {'FAIL': 1}; wall seconds 610.433; mechanism `structural`.

model_misgan_four_mode: modes=4, hq=0.99658, mass_tv=0.24414

Full scoped four-mode run completed1200updates/24checks and FAILed all5terminal complete gates. Endpoint generatorHQ=0.996582, massTV=0.244141, ambiguousHQ=0.993164, branchTV=0.044678. Preserve every terminal failed bound; no ordinary or historical100-mode pass credit.

**Next:** Stop this exact revision. The sole bounded composite masked-quantile transport plus cosine-settling followup changes the failed mass/settling mechanisms while keeping the frozen task, initializer, prior, sampling and1200update protocol.

[Record and receipt provenance](records/readout-1858b3294015f9149e7fcb9d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-186c24688ccd9e1dbc108b9d

**Scope:** research_diagnostic; scientific; revision `a1737701fadd03a96c0abbda50a967479813897f987fd7b04fb73668224e618c`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.56; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-186c24688ccd9e1dbc108b9d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v7 · readout-1b511c8eec2bb40f41aee6ec

**Scope:** research_diagnostic; scientific; revision `71405ef7cd9845970eb77c21283ef064e80ed8b099665a5b75e2a17b602cb7eb`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.261; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for empf-v7 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-1b511c8eec2bb40f41aee6ec.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-1ccb789078ba83ef238d9a80

**Scope:** research_diagnostic; scientific; revision `57f556b8cddb54cbfc93c412d824e0baa588b8e62b9ab3016e24adbae7f7b4b7`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 67.647; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-1ccb789078ba83ef238d9a80.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-1d42b97bbb8f1366a766a233

**Scope:** current; scientific; revision `97cff463de780cdb270a5cb968799379fe2eef3bbad671fd59df0c3d9515eeb0`.

Ordinary re-qualification of the unchanged selected BISPF recipe on the current develop source, reusing the compatible prior-ablation A1 evidence; the recorded discriminator_stability outcomes are reproduced.

**Observed:** {'BLOCKED': 1, 'FAIL': 4, 'INCOMPLETE': 3, 'PASS': 6}; wall seconds 1300.973; mechanism `structural`.



Ordinary discriminator_stability revision-14 re-qualification stops at Tier 1: six API model toys fail or crash, zero_tanh_smoke is unsupported, five-word smoke hit its 900 s wall clock. No new inventory row registered; revision-9 qualification retained. See reports/prior-ablation-2026-10-25/README.md.

**Next:** Repair the Tier 1 execution defects or hold the API model toys before re-qualifying.

[Record and receipt provenance](records/readout-1d42b97bbb8f1366a766a233.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v8 · readout-1e8054c27ba98f19d9d9202a

**Scope:** current; scientific; revision `a156b0664f8133339702d2c98bd8ff8e88ca912a2acfd320d2ea0a0bddb7d84e`.

Binding Feydy off-diagonal zero-distance clone deduplication strictly to discrete zero-noise ParticlePrior codebooks via rhbi_self_coupling='feydy_discrete' (alongside rhbi_simplex_weight=0.35 and rhbi_cond_ot_mode='imq_cmmd', rhbi_cond_ot_weight=1.5) preserves full quantized empirical bin multiplicity on continuous-density MoG priors—restoring both stretch precision_gaussian_mxfp8 and precision_gaussian_nvfp4 (cdf_ks <= 0.05) to exact rhbi-v6 parity—while simultaneously retaining rhbi-v7's newly unlocked passes on img_blobs4 and img_intensity2 plus img_stripes2 and all 82/115 counted view requirements under one global clock-free, shape-gate-free recipe on protocol seed 0.

**Observed:** {'FAIL': 13, 'PASS': 28}; wall seconds 2817.956; mechanism `structural`.

ae_gan_hold: recon_mse=0.0053954

rhbi-v8 passes 28/41 executed tasks (0 BLOCKED) on ordinary discriminator_stability CUDA seed-0 qualification on NVIDIA RTX A6000, confirming the study prediction on precision_gaussian_nvfp4 (cdf_ks=0.037634 <= 0.05, std_ratio=1.0183) and restoring precision_gaussian_mxfp8 (cdf_ks=0.022147, std_ratio=1.0861) while simultaneously retaining both held discrete image breakthroughs img_blobs4 (hq=0.96875, modes=4) and img_intensity2 (hq=0.90625, modes=2), img_stripes2 (hq=1.0), vector_overlap (PASS), and all 82/115 counted view requirements (27/27 Tier 1, 55/78 Tier 2).

**Next:** Promote rhbi-v8 as the canonical RHBI whole-configuration selection, register its frozen source snapshot, refresh the uncontended 60-cell DCGAN scaling table and compiled reports, and target anisotropic intra-mode covariance adaptation and 100-mode grid acquisition next.

[Record and receipt provenance](records/readout-1e8054c27ba98f19d9d9202a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-1e8d4ee06e39d85dc9988963

**Scope:** research_diagnostic; scientific; revision `d961da9d09e575e3cfc6f6c86856ea129c0c3fb68f1094053fe132b388fd06ff`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.024; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-1e8d4ee06e39d85dc9988963.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-bcap-matched-v1 · readout-21b533460c29dd5049aa64df

**Scope:** current; scientific; revision `18096544dede6a02e09e34be118d13e5554caa5aba4216980dbb442e91761312`.

Replace only the critic penalty by frozen historical bcap while retaining the K3P optimizer, noise, A2, host and sampling contracts.

**Observed:** {'FAIL': 1, 'PASS': 8}; wall seconds 86.357; mechanism `structural`.

ae_gan_hold: recon_mse=0.0052466

Fixed matched BCap passed all 3 smoke requirements and the first 5/19 quality requirements, then failed mode_hold: {'cover': 0.625, 'effective_modes': 4.520427227020264, 'hq': 0.999755859375, 'modes': 5, 'n_modes': 8, 'step': 1200}. Endurance and remaining quality tasks are unknown. Nine completed attempts cost 86.356708358 paid seconds; no execution errors or remaining reservations.

**Next:** Stop this frozen failed cohort; do not continue mode hold or infer native/shape quality from cloud behavioral passes. Investigate the saved mode-hold state under a separately bounded hypothesis before new training; calibration and policy-aware tasks remain required for adoption and broader comparisons.

[Evidence](../../reports/forge/technique-receipts/0184ca1327794ab58168d853ce0ad978.json) · [Record](records/readout-21b533460c29dd5049aa64df.json)

### atlas-particle-normal-v4 · readout-220c7c0eba270609fd9233a6

**Scope:** research_diagnostic; scientific; revision `04788d1b44fb6df84e0e451a0a1e2509213f7e5fd018afeb23a15eeb2adcdd38`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.675; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-220c7c0eba270609fd9233a6.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-23a27d9f41e3d4285248abda

**Scope:** research_diagnostic; scientific; revision `5202a22fc554b1412831075bc230f74f0fb16464f2215e167c32e0d2209f98b5`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 12}; wall seconds 124.175; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-23a27d9f41e3d4285248abda.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v9 · readout-264443fda23520da74a1fe0a

**Scope:** current; scientific; revision `9e81b5afc9298a9cea9a90952637fc73b11ca93f518c0777145ec19ce1c6efd8`.

Pairing shock-safe cyclic-midpoint convex-unimodal thermal mean gain (rhbi_cvx_mu_gain=0.58) with H_0 Vietoris-Rips connected-manifold simplicial chart alignment (rhbi_simplex_chart_weight=4.0) under one global clock-free, schedule-free, shape-gate-free recipe converts both gaussian1d_stability (cdf_ks <= 0.035, stationary + deadline reacquisition + shifted hold PASS) and stretch interp_img_patch_shift (all 5 terminal checks PASS) while preserving precision_gaussian_fp32/mxfp8/nvfp4, discrete image tasks img_blobs4/img_intensity2/img_stripes2, vector_overlap, and all prior RHBI view passes on protocol seed 0.

**Observed:** {'FAIL': 11, 'PASS': 30}; wall seconds 3577.791; mechanism `structural`.

vector_unequal_mass: hq=0.95776, mass_tv=0.092432

rhbi-v9 passes 30/41 executed tasks (0 BLOCKED, 0 regressions) on ordinary discriminator_stability CUDA seed-0 qualification on NVIDIA RTX A6000, confirming the study prediction on gaussian1d_stability (cdf_ks=0.029173 <= 0.035, mean=3.0027, std_ratio=1.0185, lifting counted view requirements to 83/115: 27/27 Tier 1, 56/78 Tier 2) and converting stretch interp_img_patch_shift (PASS, interp_peak=0.8574 >= 0.78, interp_rmse=0.0360, quadrant_coverage=4, first passer across all 10 families) while retaining precision_gaussian_fp32, precision_gaussian_mxfp8 (cdf_ks=0.02140), precision_gaussian_nvfp4 (cdf_ks=0.03290, std_ratio=1.0033), img_blobs4 (hq=0.96875), img_intensity2 (hq=0.90625), img_stripes2 (hq=1.0), vector_overlap, ring16_acquisition, and mode_hold.

**Next:** Promote rhbi-v9 as the canonical RHBI family candidate and investigate continuous-conditioned denoising transition alignment on model_ddgan_gaussian_chain, multi-bar discrete image repulsion on img_bars4, and adaptive anisotropic scale preconditioning on vector_anisotropic/unequal_width/unequal_mass.

[Record and receipt provenance](records/readout-264443fda23520da74a1fe0a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-271e0b4766487e5e23d59018

**Scope:** current; scientific; revision `4717b4b13aba2f8a4f523a5b8c66863b60d55c5d05f4bfa2dd6a238a829a86a7`.

Ordinary re-qualification of the unchanged selected r1r2 recipe on the current develop source reproduces its recorded discriminator_stability outcomes.

**Observed:** {'BLOCKED': 1, 'FAIL': 3, 'INCOMPLETE': 5, 'PASS': 5}; wall seconds 236.071; mechanism `structural`.

ring16_acquisition: modes=16, hq=0.84424, mass_tv=0.051514

Ordinary discriminator_stability revision-14 re-qualification reproduces all 7 core Tier 1 verdicts (5 PASS, ring16_acquisition FAIL, clockfree_audit_measurement_v1 FAIL) and stops at Tier 1; revision-9 qualification retained.

**Next:** Retain revision-9 qualification in technique-inventory; address ring16_acquisition and clockfree_audit_measurement_v1 in family follow-up.

[Record and receipt provenance](records/readout-271e0b4766487e5e23d59018.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-no-critic-penalty · readout-28f0c9140b4cfeef57f23dd2

**Scope:** current; scientific; revision `aa9e200be9f520bf97e8d3408829b852188c80e2b64aa6f02430f58eaec117c9`.

Removing the public critic gradient penalty tests whether cheap stability gates reject a substantively weakened formulation before quality work.

**Observed:** {'FAIL': 1}; wall seconds 7.827; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 1.2546037435531616, 'mean_abs': 0.4477476179599762}. One completed attempt cost 7.827370992 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/33bc62d63509425e9f7237ef19ee7374.json) · [Record](records/readout-28f0c9140b4cfeef57f23dd2.json)

### r1r2-repair-control-20261022 · readout-2b96c09667e7a4aef944ce96

**Scope:** research_diagnostic; scientific; revision `3cca6de5d931b16469b0ef0fedbc0b5cb21f55b8af202a7868e94277fcef2037`.

This exact source-bound global recipe can retain its own earliest independently confirmed word acquisition through every original scheduled primary and independent confirmation check, including the exact restored state, for4000 further updates. The original word_hold evaluator and all25 paired checks are authoritative; endpoint token probability alone cannot establish retention. This diagnostic changes scheduler placement only and grants no ordinary qualification.

**Observed:** {'PASS': 2}; wall seconds 158.69; mechanism `structural`.

five_word_joint_hold: modes=5, mass_tv=0.020898; five_word_joint_smoke: modes=5, mass_tv=0.018945

Own-checkpoint acquisition and full4000-update hold PASS: all25 scheduled/restored-state checks confirmed, parent update834, endpoint minimum token probability0.9979230761528015. No ordinary qualification credit. The NS/AMSGrad reference already passes retention; explicit reconstruction improves confidence margin but is not necessary for these gates.

**Next:** Close the finite three-role campaign. Keep R2; do not promote the failed global repairs. Preserve all ordinary failures and archived source identities. Require a demonstrated density or conditional-field prerequisite before another candidate.

[Evidence](../../reports/forge/r1r2-repair-20261022/README.md) · [Record](records/readout-2b96c09667e7a4aef944ce96.json)

### k3p · readout-2c2eb24207b627b2ad26ea18

**Scope:** calibration_diagnostic; scientific; revision `5e4a04632539a2ae1fa6d021e54f2ab5172245e7a028c795d45bfff9c2284426`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 2, 'PASS': 3}; wall seconds 156.284; mechanism `structural`.

mode_hold: modes=5, hq=0.99878; img_intensity2_residual16: modes=2, hq=0.90625; img_bars4_residual16: modes=4, hq=1

Joined host-profile diagnostic: 3 PASS and 2 FAIL across five cells for 156.283765995 paid seconds, zero execution errors/reservations. Bars4, intensity and unequal mass pass. Mode-hold fails 5/8 modes; native grid loses shape accuracy after acquiring it and fails terminal/100k holdout gates. Zero unintended RNG deviations; frozen grades independently reproduced. Exact revision concluded.

**Next:** Preserve this control and do not fill its remaining quality cells or extend the failed 7k parent. A separately registered A2-off native diagnostic changes one substantive factor while preserving seed/init/MoG width and full gates; reuse this certified grid result without another control launch. Do not infer a cause or a positive lineage from this single run.

[Record and receipt provenance](records/readout-2c2eb24207b627b2ad26ea18.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-onboarding-anchor-ablation · readout-2c30cecb0a3f37d001ce3b28

**Scope:** calibration_diagnostic; scientific; revision `f3adbc07708643ba4563a65ce256b6e1903e30a90095032c6821118cbb36e775`.

Removing the public critic EMA-anchor term may weaken short-horizon adversarial movement; the fixed CPU smoke gate should reject a harmful ablation before larger tests.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 17.008; mechanism `structural`.

ae_gan_hold: recon_mse=0.004509

Corrected-source smoke matrix: two_pole FAIL, unused_token_hold PASS, ae_gan_hold PASS. Whole-screen FAIL; independent reference outcomes remain unmeasured. All selected attempts completed, with actual named-stream seed labels now consistent.

**Next:** Keep ordinary qualification stopped and retain this frozen cohort. The two-cell learned-MoG GPU pilot is registered but must wait for the requested joint device window. Collect deeper independent references only through explicit diagnostic selections, then evaluate the unchanged calibration criteria before adoption.

[Record and receipt provenance](records/readout-2c30cecb0a3f37d001ce3b28.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v3 · readout-2e0a16b47298ebbc5e33ba63

**Scope:** current; scientific; revision `1e0d674047a1212d8a18623dac1886338156658cd4808769641f83594adef0bc`.

Adding stateful batch-view detection in RHBICriticPenalty and fibre-centered conditional relativistic slack routing (rhbi_conditional_fibre='slack') to rhbi-v2 preserves all 11/11 Tier 1 passes (27/27 across views) and all 7 rhbi-v2 Tier 2 passes while repairing trajectory, residual_student, and model_ae_ddgan_bimodal_chain under one unchanged global clock-free recipe on protocol seed 0.

**Observed:** {'FAIL': 17, 'PASS': 21}; wall seconds 2567.504; mechanism `structural`.



prediction_observed

**Next:** review_saved_diagnostics

[Record and receipt provenance](records/readout-2e0a16b47298ebbc5e33ba63.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v8 · readout-30ab63b327f728a298409e42

**Scope:** research_diagnostic; scientific; revision `a156b0664f8133339702d2c98bd8ff8e88ca912a2acfd320d2ea0a0bddb7d84e`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 71.785; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for rhbi-v8 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-30ab63b327f728a298409e42.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-31599cfd98b712a799a0ae92

**Scope:** research_diagnostic; scientific; revision `41c9a8b9a9f4a518a140d37072849826ab197defcb2ce39cbf64943aacfd09d2`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 64.113; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-31599cfd98b712a799a0ae92.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-31dc4212b265fe25bad81d5d

**Scope:** research_diagnostic; scientific; revision `21dc1f345268686aae9f13f39127e12d0250b3921bc6217d63b65c6d5fb3103e`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 64.187; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-31dc4212b265fe25bad81d5d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-322bc05e8bb593d80d65cb36

**Scope:** research_diagnostic; scientific; revision `fbc1bf4bda35c195efe7830c359d1851febdbca85b4257b4415d66187ed3d248`.

Measure update throughput and preserve the sustained stripe PASS on unchanged small-image task contracts. Retain the held bars outcome.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 55.034; mechanism `structural`.

img_bars4: modes=2, hq=0.96875; img_stripes2: modes=2, hq=1

Original small-image controls retain stripes PASS and bars FAIL at the unchanged 600-update budget.

**Next:** Preserve these exact controls and do not repeat unchanged experiments.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-322bc05e8bb593d80d65cb36.json)

### recipe-prior-refactor-r1r2-v1 · readout-32a1753d5c69888caf94e9d0

**Scope:** research_diagnostic; scientific; revision `9ef98b984e75f7e919b98af7b89040eabbcb8171c2496cfa7a8f66179bd341de`.

Evaluate whether unchanged selected r1r2 recipe (recipe-prior-refactor-r1r2-v1) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 56.205; mechanism `structural`.



Selected r1r2 recipe passes interp_img_patch_shift (5/5 terminal checks: interp_rmse=0.0481, interp_peak=0.8876, midpoint_novelty=0.5210), puts 12/12 endpoints on sparse_arc (endpoint_hq=1.000, 28/32 bins) with secant chord cutting (interp_hq=0.5667), and suffers a terminal step-800 spike on saddle_barycentric.

**Next:** Stabilize late-horizon zero-sigma R1/R2 critic penalties on 3D curved surfaces and regularize interior MLP chord curvature.

[Record and receipt provenance](records/readout-32a1753d5c69888caf94e9d0.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-36bc933bc99f88a6f66d630d

**Scope:** research_diagnostic; scientific; revision `f71aad6445dd379483c9d37c262adb80c87396af2599080096a18c31c94ac8a9`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 165.368; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-36bc933bc99f88a6f66d630d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-v2 · readout-3721266a32807c156725dfc8

**Scope:** research_diagnostic; scientific; revision `34571ac88b53ec2ad8c9dead16a5b23ba3bb408dcd2f2cf3d8c6d8c290d93eef`.

Measure incumbent native BISPF on explicit width64 architecture-portability variants of stripes2 and bars4. These new larger-capacity task identities retain the original target, numerical gates, prior and 600-update budget and provide the actual native control for a later matched optimized solver comparison.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 145.379; mechanism `structural`.

img_bars4_w64_optimizer_diagnostic: modes=4, hq=0.90625; img_stripes2_w64_optimizer_diagnostic: modes=2, hq=1

Native BISPF on the two explicitly wider architecture-portability hosts passes stripes (HQ1,2 modes,14 terminal passing checks) and fails bars (endpoint HQ0.90625,4 modes but only2 terminal passing checks versus5 required). Both arms complete600 updates and24 observations with finite states and seed0 public initialization.

**Next:** Retain both native outcomes as controls. Quintic passes both wider tasks and proceeds to separately frozen full performance and ordinary quality; Frobenius regresses stripes and stops. No default adoption follows.

[Record and receipt provenance](records/readout-3721266a32807c156725dfc8.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-38c1b334ef586a06a89392b7

**Scope:** research_diagnostic; scientific; revision `ad1a2e99d28e1485a7a33510e1e5bbd503e1d760fef45af1964e12e6d0ca42ef`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 169.508; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-38c1b334ef586a06a89392b7.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2-repair-r1cycle-20261022 · readout-39165e761d4e5ea74a5a67d1

**Scope:** current; scientific; revision `de3afa5bb4c97c9e51a2ef66817108ca2372525aebff8200537f9df9d3328421`.

The single global R1-only plus explicit joint inverse reconstruction recipe repairs the unchanged Ring16 gate through its R1 critic change while retaining confident paired word acquisition with a stable raw-logit inverse signal and every other required Tier1 pass. This new-source operational repair fixes snapshot provider dependencies, preserves original runtime errors and routes the already-declared tanh evaluator; all original scientific gates and recipe values remain unchanged.

**Observed:** {'FAIL': 4, 'PASS': 10}; wall seconds 379.634; mechanism `structural`.

ae_gan_hold: recon_mse=0.0084931; five_word_joint_smoke: modes=5, mass_tv=0.018945

Word confidence signature observed, but global repair hypothesis fails: 9/13 required PASS, 4 FAIL; separate clock PASS. Word first confirmation returns to 834 and endpoint minimum token probability rises to .9999932051. All nonword numerical trajectories match R1-only; Ring, DDGAN, AE-DDGAN and AE-GAN failures remain.

**Next:** Retain R2 and the selected historical recipe; no ordinary Tier2 or default promotion. Finish separately scoped own-checkpoint word hold, publish all failures, and require a demonstrated density-force/conditional-tracking prerequisite before another global candidate.

[Evidence](../../reports/forge/r1r2-repair-20261022/README.md) · [Record](records/readout-39165e761d4e5ea74a5a67d1.json)

### recipe-prior-refactor-ka2-v1 · readout-3ac6c4c08f886f05d0b92dfc

**Scope:** research_diagnostic; scientific; revision `d374199b825ad11bf123d47e7b7d68eba029b49ac58e97614c49f14862198f95`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 12}; wall seconds 122.027; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-3ac6c4c08f886f05d0b92dfc.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-3b448c78f80dda9bc15dc448

**Scope:** research_diagnostic; scientific; revision `753c4da3d2b446d4b6f0486e060466b48a250398fa053789b5d8dcdb574d2afc`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 66.235; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-3b448c78f80dda9bc15dc448.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-3bd45e331f5dd03701da2700

**Scope:** current; scientific; revision `5e8301e5d868022f59e92f0e0c0e67348bdebd00d372ace75f998abe7d469fad`.

Ordinary re-qualification of the unchanged selected release07-gan-v3 recipe (recipe-prior-refactor-release07-gan-v3-v1) on current develop source 2e8a97a0aa84 under protocol seed 0 on NVIDIA RTX A6000 records current-source discriminator_stability and shared Tier 1 goal-view verdicts.

**Observed:** {'FAIL': 4, 'PASS': 10}; wall seconds 389.801; mechanism `structural`.



Completed discriminator_stability seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (FAIL 4, PASS 10). Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact.

**Next:** Retain source-bound readout and compiled Forge reports.

[Evidence](../../) · [Record](records/readout-3bd45e331f5dd03701da2700.json)

### recipe-prior-refactor-r1r2-v1 · readout-3e30803cb8dcd9b27ac20c87

**Scope:** research_diagnostic; scientific; revision `9e2c39393e23019568d0e93585a7d90fb55c457a371be8576365fffed4b03b5c`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.574; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-3e30803cb8dcd9b27ac20c87.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-v1 · readout-3f3f893ae724d651f1655fc9

**Scope:** current; scientific; revision `45e5304bd6d19af271e0d2072cf1c5d36a3cf8cbcbf45e9eada0218c8ad0782b`.

Born-Infeld Sinkhorn Particle Flow (critic_formulation=bispf, prior_regularizer=sbnp, prior_reg=0.01) passes all 6 Tier 1 gates, repairs mode_hold in Tier 2 while preserving all 9 BCAP leader Tier 2 passes (including five_word_joint_hold), and passes both Tier 3 endurance gates (ring_hold and ring_extension) under constant learning rates.

**Observed:** {'FAIL': 11, 'PASS': 17}; wall seconds 3479.478; mechanism `structural`.

img_stripes2: modes=2, hq=1; five_word_joint_smoke: modes=5, mass_tv=0.018945

Born-Infeld Sinkhorn Particle Flow (BISPF, bispf-v1) passes 6/6 required Tier 1 tasks plus clockfree_audit_measurement_v1 (7/7 total, 22/22 across all 6 views) and 10/21 required Tier 2 tasks (55/116 across all 6 views, 77/152 total across views vs. 71/152 for the BCAP DualNorm leader). BISPF repairs mode_hold (FAIL -> PASS: modes=8, effective_modes=7.47686, cover=1.0, hq=0.99951) while preserving 25/25 mandatory checks on five_word_joint_hold and all 8 other BCAP leader Tier 2 passes (0 regressions), and improves 100-mode grid100 precision from 0.25801 to 0.37015 (+43.5% relative) and staggered100 precision from 0.29954 to 0.34184 (+14.1% relative). Tier 3 remains blocked by the 11 remaining Tier 2 failures.

**Next:** Retain bispf-v1 as the selected current measurement for the BISPF family (16/29 on discriminator_stability, 77/152 across views). Target the remaining 11 Tier 2 bottlenecks (gaussian1d_stability post-shift transport, non-uniform mass/anisotropic 2D mixtures, 4-mode conv image symmetry breaking, and 100-mode cluster tightness) in a bounded follow-up study before spending on Tier 3.

[Evidence](../../reports/forge/bispf/README.md) · [Record](records/readout-3f3f893ae724d651f1655fc9.json)

### recipe-prior-refactor-release07-gan-v3-v1 · readout-40767bea377ac807d62a047a

**Scope:** research_diagnostic; scientific; revision `af91a2d7a18e4a328dc19baedc201f2d690a135aff5df6ea7a9d59cb29c9e7ca`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.293; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/release07-gan-v3-dcgan_performance_b64-01718bbf7d27ee3b.json) · [Record](records/readout-40767bea377ac807d62a047a.json)

### recipe-prior-refactor-bcap-pure-v1 · readout-4197ba117621ba0fe1dd2c1c

**Scope:** research_diagnostic; scientific; revision `51374ebe946901607490e463ad90799aa59fe9223b0d7be2d24fd3a5c844dcb2`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 165.333; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-4197ba117621ba0fe1dd2c1c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-423f82f13ee30c0a95fd266a

**Scope:** research_diagnostic; scientific; revision `efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd`.

Guarding LossSurprise.decide against KA2 pre-anchor critic-only SettledReopenGuard witnesses (guard.ka2_anchor_started is False and witness_roles == {'critic'}, where the critic ladder has stepped down during Phase A acquisition while the generator ladder remains unsettled at full rate s=1.0) prevents premature step-35 critic re-empowerment on N=12 mode_hold while preserving post-anchor equilibrium shift detection, recovering 8/8 mode_hold and unblocking ring_hold, ring_extension, and target_shift_recovery while retaining birth_death_cells=256 and NormalNoiseManipulator(sigma=0.0065) to preserve all 6/6 vector, 4/4 image, 2/2 VAE, and 2/2 clockfree passes for >= 24 PASS / 0 INCOMPLETE with zero task regressions relative to both atlas-particle-normal-v2 and atlas-particle-normal-v3 under protocol seed 0.

**Observed:** {'FAIL': 14, 'PASS': 24}; wall seconds 4407.023; mechanism `structural`.

img_blobs4_residual16: modes=4, hq=0.96875; img_intensity2: modes=2, hq=1

Guarding LossSurprise.decide against KA2 pre-anchor critic-only SettledReopenGuard witnesses (guard.ka2_anchor_started is False and witness_roles == {'critic'}) prevented the premature step-35 critic reopen on N=12 mode_hold while preserving post-anchor equilibrium shift detection across CPU and CUDA, recovering mode_hold (8/8 modes, hq=0.9998) and unblocking ring_hold, ring_extension, and target_shift_recovery for 24 PASS, 14 FAIL, 0 INCOMPLETE across all 38 executable tasks (6 prerequisite-deferred, 4 shared_pending unrouted adapters) with zero task regressions relative to both atlas-particle-normal-v2 and atlas-particle-normal-v3.

**Next:** Investigate adaptive per-population cell budgeting so N=256 rare-mode vector hosts avoid under-resolved k=7 kNN turnover without doubling K=256 FeatureCellBirthDeath cells on N=20000 100-mode tasks (grid100, rotated100, staggered100), and investigate joint-inference stability for BiGAN/DDGAN/five_word_joint hosts.

[Record and receipt provenance](records/readout-423f82f13ee30c0a95fd266a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v11 · readout-4301e02799ec0ad34cc958a1

**Scope:** current; scientific; revision `05f71faf03f06323d92bc7def373a5310d32c91bea38bf62b760725e29415f55`.

Moderating the continuous-prior valley 96th-quantile precision thermal residual gain to rhbi_tail_spill_gain=0.95 alongside wide-span H_0-connected macro-simplicial settling (rhbi_simplex_macro_settle=true) under one global clock-free, schedule-free, shape-gate-free recipe restores full terminal stability on Tier 1 model_ae_gan_multimodal while converting both counted vector_anisotropic (component_covariance_error <= 0.85, sw1_normalized <= 0.18, mass_tv <= 0.15, hq >= 0.85) and stretch interp_saddle_barycentric (all 5 terminal checks PASS) with zero regressions against rhbi-v9 on protocol seed 0.

**Observed:** {'FAIL': 9, 'PASS': 32}; wall seconds 3207.114; mechanism `structural`.



confirmed_positive: rhbi-v11 passed 32/41 executed discriminator_stability tasks (11/11 required + 2/2 diagnostic Tier 1 PASS = 27/27 Tier 1 view requirements, 12/18 required Tier 2 PASS = 61/78 Tier 2 view requirements, 88/115 total counted view requirements, 0 BLOCKED) on protocol seed 0, converting both counted vector_anisotropic (component_covariance_error=0.4987 <= 0.85, sw1_normalized=0.1482 <= 0.18, mass_tv=0.1449 <= 0.15, hq=0.9468 >= 0.85) and stretch interp_saddle_barycentric (barycenter_hq=0.8358 >= 0.78, barycenter_surface_mae=0.0668 <= 0.09, endpoint_surface_mae=0.0399 <= 0.08) to PASS with zero regressions.

**Next:** Promote rhbi-v11 as the canonical RHBI family selection, refresh uncontended 60-cell DCGAN scaling and full-suite readouts, and target remaining discrete-codebook dense lattice coverage (grid100/rotated100/staggered100) and sparse arc interpolation (interp_sparse_arc) in future research.

[Record and receipt provenance](records/readout-4301e02799ec0ad34cc958a1.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-44227b0a00ae4b4b0c510872

**Scope:** research_diagnostic; scientific; revision `b9ebd32a3d1a721670787e564346134bf6ddb2b668492960e842fa728e49c616`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.692; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-44227b0a00ae4b4b0c510872.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v2 · readout-44a71042e8d9d2018cfd677e

**Scope:** current; scientific; revision `2f078423d65494cd5a7f7c15210df7d3719fa6a73ab12564142adb0b529839e2`.

Replacing logarithmic rapidity screening and hollow self-coupling in rhbi-v1 with linear Born-Infeld screening (rho=3.0), Feydy diagonal self-coupling, endpoint slope floor damping (0.005), SBNP prior regularization (0.02, prior_lr_mult=2.9), lr=0.014, and polar smoothing 0.0061 passes all 11 required Tier 1 discriminator_stability gates on protocol seed 0 and unlocks Tier 2 and Tier 3 measurement.

**Observed:** {'FAIL': 17, 'INCOMPLETE': 2, 'PASS': 19}; wall seconds 2737.438; mechanism `structural`.

five_word_joint_hold: modes=5, mass_tv=0.03457; vector_unequal_mass: hq=0.98389, mass_tv=0.082207

incomplete

**Next:** Declare rhbi-v3 fixing _BoundRHBIPenalty wrapper concat, continuous-fibre conditional screening in RHBILoss, and mode_hold retention.

[Record and receipt provenance](records/readout-44a71042e8d9d2018cfd677e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-44a844de0b6e23e430a12858

**Scope:** research_diagnostic; scientific; revision `52bcfbda682fdbe578d1c0c352a31673659d5679d685e8aa4d21931940790e56`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.564; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-44a844de0b6e23e430a12858.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-454e20c82f89cc56488e7f98

**Scope:** research_diagnostic; scientific; revision `c2c710554730bb558077ac74b96b10fc306098725e2004abe1b5c91362c5f6f4`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 82.367; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/e22-dcgan_performance_b64-82b8a8a97ca180b5.json) · [Record](records/readout-454e20c82f89cc56488e7f98.json)

### ka2--8d6bb101262f49dcef37111b37ab781ca09652b47ad95e783ccea4dfbec680b4 · readout-457f93f65c1e6782fba74306

**Scope:** current; scientific; revision `8ff7b28082062dbb985e5b9ceb2f5cda878c7afc57ca55b9b00221e5dd2e80e3`.

Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.

**Observed:** {'FAIL': 1}; wall seconds 5.994; mechanism `floor_constant`.



This exact global configuration failed the first two_pole movement gate. Tier1 is0/5; word and25remainingcells are UNKNOWN. No configured standard or default adoption.

**Next:** Bounded global search concluded. Retain the exact historical family incumbent as unqualified evidence. No further paid stage is authorized; require a separate bounded global proposal before any continuation.

[Record and receipt provenance](records/readout-457f93f65c1e6782fba74306.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-464567d128031fee54c42252

**Scope:** research_diagnostic; scientific; revision `fbc1bf4bda35c195efe7830c359d1851febdbca85b4257b4415d66187ed3d248`.

Matched original-quintic control for the throughput recipe on the two unchanged wide-image tasks, comparing the exercised precision and optimizer execution changes.

**Observed:** {'PASS': 2}; wall seconds 45.999; mechanism `structural`.

img_bars4_w64_optimizer_diagnostic: modes=4, hq=1; img_stripes2_w64_optimizer_diagnostic: modes=2, hq=1

Both unchanged width 64 image controls PASS their complete 600-update sustained gates.

**Next:** Retain the source-bound controls and actual-training GIFs; do not rerun unchanged arms.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-464567d128031fee54c42252.json)

### bispf-optimized-v3 · readout-47d84c98f42853a07d20becb

**Scope:** current; scientific; revision `e38fbbdd10585f61ef6ff2a694e2087746956511dff55afa441fbb63497267c6`.

The frozen CUDA-graph augmented-polar BISPF optimizer version 3 reduces DCGAN update latency while preserving every incumbent counted task pass: all six Tier 1 tasks, all ten counted Tier 2 tasks, the counted ring endurance verdict, and the separate clock and ring-hold diagnostic passes. The complete unchanged held task set remains measured and visible; no original result is replaced or inherited.

**Observed:** {'PASS': 1}; wall seconds 23.453; mechanism `structural`.



Stop this numerical-backend revision: graph acceleration improves the 64-pixel profile but misses the declared 256-pixel target. The ordinary quality study was started prematurely and canceled after one completed smoke pass, giving no full qualification.

**Next:** Retain the rejected solver and partial costs under their original identities. Test separately declared optimizer-rule alternatives with direct wide-channel controls before an ordinary full study or selection.

[Record and receipt provenance](records/readout-47d84c98f42853a07d20becb.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v7 · readout-4901963885d977da23e139ba

**Scope:** current; scientific; revision `e9cb4e9e9d26179d1fa70cf52b42cee045c9c8e67238b78e5c0c28ac0d8fda8e`.

Size-gating DualNorm's guarded cubic Newton-Schulz dense polar profile to min_dim >= 64 and max_dim > 128 (optimizer_dense_profile = 'newton_schulz_hybrid_v1') preserves exact polar_factor SVD trajectories across all hidden <= 128 tasks (including model_ae_gan_multimodal and precision_gaussian_fp32) while eliminating 78.8% of cuSOLVER SVD latency on five_word_joint_smoke so empf-v7 clears all 13/13 counted Tier 1 gates and advances through Tier 2/3.

**Observed:** {'FAIL': 15, 'PASS': 23}; wall seconds 3973.882; mechanism `structural`.

ae_gan_hold: recon_mse=0.0043931; five_word_joint_hold: modes=5, mass_tv=0.03457

empf-v7 cleared all 13/13 counted Tier 1 gates (including five_word_joint_smoke in 608.9s, model_ae_gan_multimodal, model_vae_categorical, model_vae_hard, and precision_gaussian_fp32) and confirmed its study prediction on gaussian1d_stability (PASS, final cdf_ks <= 0.05), reaching 23 PASS on the 38-task discriminator_stability view (up from 22 PASS in empf-v2).

**Next:** Promote empf-v7 as the selected canonical EMPF candidate in family-current-v1.json and trainer-families.json, and run full Forge readout refresh across all 9 views and scaling benchmarks.

[Record and receipt provenance](records/readout-4901963885d977da23e139ba.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-vaegan-settle-v1 · readout-4a3861ac6f4b3a836975d087

**Scope:** research_diagnostic; scientific; revision `15cec064bfab5931854da2b176eb21f34230e1508407767ad76ab3f2fc721660`.

A single global finite-budget cosine settling schedule repairs both unchanged VAEGAN tasks after transport-only failed; retain all adversarial, reconstruction and latent regularization objectives. This schedule comparison does not prove transport necessary.

**Observed:** {'PASS': 2}; wall seconds 445.906; mechanism `floor_constant`.



One global BISPF transport-plus-cosine-settling recipe passes both full1200-update VAEGAN tasks and all5 terminal checks, with hard/categorical passing suffixes6/9. Actual KL, input-dependent reconstruction and both independent sampling laws satisfy every unchanged bound.

**Next:** Stop after6runs, retain all4 failures and2 passes. Recommend this recipe for these two declared toys; next separate causal question is schedule-only without transport. No extra seed run, continuation or promotion.

[Record and receipt provenance](records/readout-4a3861ac6f4b3a836975d087.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v6 · readout-4aa764495c669b9247ca874b

**Scope:** current; scientific; revision `2929a9ef553dde2092e0769ebdeaa03d0f6ded4ae74c07482ab71a77c601abe7`.

Replacing serial 2D cuSOLVER SVD calls with DualNorm's guarded cubic Newton-Schulz regularized polar profile (optimizer_dense_profile = 'newton_schulz_v1') eliminates multi-agent GPU cuSOLVER contention on five_word_joint_smoke so empf-v6 clears all 13/13 counted Tier 1 gates within wall-time budgets and advances through Tier 2/3 to realize gaussian1d_stability, target_shift_recovery, ring_hold, and ring_extension gains.

**Observed:** {'FAIL': 3, 'PASS': 12}; wall seconds 1170.146; mechanism `structural`.



empf-v6 cleared five_word_joint_smoke in 549.7s (20,001/20,001 steps, modes=5, quality_fraction=1.0, mass_tv=0.0189) and passed 11/13 counted Tier 1 gates, but applying 32-iteration cubic Newton-Schulz to small dense matrices (hidden <= 64) regressed model_ae_gan_multimodal (reconstruction_mse 0.0405 vs 0.0080) and precision_gaussian_fp32 (step 834 cdf_ks 0.0509 vs <= 0.05).

**Next:** Define empf-v7 with a size-gated hybrid dense profile (newton_schulz_hybrid_v1) mirroring DualNorm's per-offset convolution rule: exact rank-1 normalization for vector-shaped 2D heads, exact polar_factor SVD for small matrices (min_dim < 64 or max_dim < 128), and guarded cubic Newton-Schulz for large dense matrices (min_dim >= 64 and max_dim >= 128).

[Record and receipt provenance](records/readout-4aa764495c669b9247ca874b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-4bd94051f203bbbb28a3e7a6

**Scope:** current; scientific; revision `87d849138dd0ae385260a77223e9a70a7c2a8a112733b2312f556b462cc164e9`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 1}; wall seconds 5.255; mechanism `structural`.



The first Forge CPU smoke pilot failed two_pole after 80 updates: mean_abs 0.10652 versus the required 0.30; grad_med 0.06372 passed. Independent frozen-source grading stopped all later work.

**Next:** Preserve this exact negative result, investigate smoke transfer predictiveness in Phase D, and continue software/multi-GPU verification without weakening this gate to turn the result into a pass.

[Record and receipt provenance](records/readout-4bd94051f203bbbb28a3e7a6.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v2 · readout-4dc982d111099c9b0f33a18e

**Scope:** research_diagnostic; scientific; revision `de4379506c30e147385182dfd8cdc61133024f4636c2ecbf07cea1c509cc2cc0`.

Re-executing atlas-particle-normal-v2 on the repaired source tree (with profile-scoped AE task snapshot sources, restored UpdatePolicy conditional row-semantics validation, restored Recipe optimizer parameter deduplication, and SettleTest/FeatureCellBirthDeath diagnostic step-stamp sanitization) reproduces >= 22 PASS / 0 INCOMPLETE on prior-ablation-full-suite-v1 under protocol seed 0.

**Observed:** {'FAIL': 16, 'PASS': 22}; wall seconds 3278.971; mechanism `structural`.

img_stripes2_residual16: modes=2, hq=0.96875

Re-verified atlas-particle-normal-v2 on the repaired source tree (profile-scoped AE task snapshot sources, restored UpdatePolicy conditional row-semantics validation, restored Recipe optimizer parameter deduplication, clean non-policy transfer_behavior serve_average/capability bindings, and active SettleTest/FeatureCellBirthDeath diagnostic step-stamp sanitization). Deterministically achieves 22 PASS, 16 FAIL, 0 INCOMPLETE across 38 executed tasks on prior-ablation-full-suite-v1 under protocol seed 0 (NVIDIA RTX A6000), observing clockfree_audit parity_comparisons=4 >= 4 and avoiding gaussian1d_smoke falsifier.

**Next:** Address 100-mode clean-covariance deficits (grid100/staggered100/rotated100) and non-schedule joint BiGAN critic balance on five_word_joint_smoke in a dedicated follow-up mechanism study.

[Record and receipt provenance](records/readout-4dc982d111099c9b0f33a18e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-no-critic-penalty · readout-4e04059eb9f631a2e406447e

**Scope:** current; scientific; revision `ef50ff6fb20723ab9340fbb8d1a6c8abd45cb2e2950250578a13dd36207113b5`.

Removing the public critic gradient penalty tests whether cheap stability gates reject a substantively weakened formulation before quality work.

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 18.799; mechanism `structural`.

ae_gan_hold: recon_mse=0.004034

The ordinary fresh-checkout two_pole attempt failed its slope bound (grad_med1.2546037>1 despite mean_abs0.4477476>=0.30). The two subsequent registered diagnostics completed: unused_token_hold FAIL and learned-MoG ae_gan_hold PASS. The whole smoke decision remains FAIL; these diagnostics cannot qualify the candidate.

**Next:** Retain all nine cheap-cell outcomes and costs. Keep ordinary qualification stopped. Review the exact historical/public-host parity differences before deeper reference spending; reserve the required GPU window and register only justified missing reference cells. Do not tune thresholds or treat unmeasured reference labels as failures.

[Record and receipt provenance](records/readout-4e04059eb9f631a2e406447e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p--370894f9284090fc9771145a869f2d7fa617283c3736c8e461a47553c9668ce6 · readout-4e44570c882c7d07bbddc813

**Scope:** current; scientific; revision `1a9eb1831acf7de8a0a96d1a2db0fc239b9836d19cfaec2986937cb0cd00e7ed`.

Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.

**Observed:** {'FAIL': 1}; wall seconds 8.084; mechanism `floor_constant`.



This exact global configuration failed the first two_pole movement gate. Tier1 is0/5; word and25remainingcells are UNKNOWN. No configured standard or default adoption.

**Next:** Bounded global search concluded. Retain the exact historical family incumbent as unqualified evidence. No further paid stage is authorized; require a separate bounded global proposal before any continuation.

[Record and receipt provenance](records/readout-4e44570c882c7d07bbddc813.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-4f82f909f64b3d97917d6b1a

**Scope:** current; scientific; revision `9e38108414dd84e7f5374086d86b08764ecc4553fbd329768438e9b71c900194`.

Ordinary re-qualification of the unchanged selected k3p recipe on the current develop source reproduces its recorded discriminator_stability outcomes.

**Observed:** {'BLOCKED': 1, 'FAIL': 2, 'INCOMPLETE': 4, 'PASS': 5}; wall seconds 251.421; mechanism `structural`.

five_word_joint_smoke: modes=5, mass_tv=0.018945

Ordinary discriminator_stability revision-14 re-qualification reproduces all 7 core Tier 1 verdicts (5 PASS, ring16_acquisition FAIL, clockfree_audit_measurement_v1 FAIL) and stops at Tier 1; revision-9 qualification retained.

**Next:** Retain revision-9 qualification in technique-inventory; address ring16_acquisition and clockfree_audit_measurement_v1 in family follow-up.

[Record and receipt provenance](records/readout-4f82f909f64b3d97917d6b1a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-5449c98aa6ac41a9138a74e6

**Scope:** research_diagnostic; scientific; revision `4c33fa871d84961a929bbfd21b0f147320e1bf8ccfb5c844792e03d19cd7d834`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.253; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-5449c98aa6ac41a9138a74e6.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-5456d16705656a42faaaedb1

**Scope:** research_diagnostic; scientific; revision `5e805792d07d8e2af59aa8d242fa7284584d55f3cc248943ed725529d5853380`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 163.169; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-5456d16705656a42faaaedb1.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-onboarding-anchor-ablation · readout-5470a57cf1a74abc896c8a55

**Scope:** calibration_diagnostic; scientific; revision `d297bd4d9cf9c012b927c8e10af4c5145a4e73aee1680a675396f99aa7cf048f`.

Removing the public critic EMA-anchor term may weaken short-horizon adversarial movement; the fixed CPU smoke gate should reject a harmful ablation before larger tests.

**Observed:** {'FAIL': 1}; wall seconds 17.57; mechanism `structural`.

mode_hold: modes=5, hq=0.99292

Registered mode-hold diagnostic FAIL: 5/8 modes, HQ .992919921875, zero passing observations in 24 checks; 1,200 complete updates and 17.569934434 paid seconds. No infrastructure error. Independent reference outcomes remain unknown.

**Next:** Conclude this mode-only diagnostic. Preserve the negative in memory; do not fill this exact calibration profile solely for adoption, since all three smoke predicates fail and the unchanged criteria require a positive with no false rejection. A new justified profile or independently compatible positive reference is needed; no general downstream or universal algorithm claim.

[Record and receipt provenance](records/readout-5470a57cf1a74abc896c8a55.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-560cdd3e8aa983df09f28d07

**Scope:** research_diagnostic; scientific; revision `c2e27da4b3f1e995e557bf479b5b599db07a2367e462593154cf5033715001e8`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.527; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-560cdd3e8aa983df09f28d07.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### ka2 · readout-56577544c52c17aa60639d19

**Scope:** current; scientific; revision `2a6cb4b3f19e57bb16599e23df1a83750b1e992634c6428dacd711f0269c83d1`.

The current KA2 public recipe is an explicit learned-MoG reference; archived K3P and E22 cloud results remain separate evidence.

**Observed:** {'FAIL': 1}; wall seconds 5.46; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 0.049373187124729156, 'mean_abs': 0.11137516051530838}. One completed attempt cost 5.459934227 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/0142b29ed9154d9fbf065c2934545c0c.json) · [Record](records/readout-56577544c52c17aa60639d19.json)

### recipe-prior-refactor-k3p-v1 · readout-56fcc7abc4b7e69d88a5c110

**Scope:** research_diagnostic; scientific; revision `63a58bbb1cc4a6f47d6b4caa97f86c8555a4bbc0150ae55a005e2fbecfa87be5`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.586; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-56fcc7abc4b7e69d88a5c110.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-a2-off-native-diagnostic · readout-5830d399b6741ce7b3a2b992

**Scope:** calibration_diagnostic; scientific; revision `8493b15ff2d0f677de94a07cf344762a1d9efd58c609987fff7dab311cad8581`.

Diagnostic hypothesis: removing the active A2 sparse-row damping may change the late covariance undershoot on the unchanged named affine grid100 learned-MoG host. This isolates one mechanism; saved evidence does not predict a positive result, and no full-reference or adoption claim is made.

**Observed:** {'FAIL': 1}; wall seconds 85.368; mechanism `structural`.

grid100_affine_square_named_v1: precision=0.99739, center_rms_sigma=0.12509, mass_tv=0.23354, radial_ks=0.21077

Reject the one-factor A2-off diagnostic on the named affine grid100 learned-MoG host. FAIL after all 7000 updates for 85.367950892 paid seconds, no execution errors or reservations. It never passed recorded coverage/accuracy; final modes74 vs baseline100, holdout massTV.23354 vs.032, covariance bias-.426600 vs-.265594, radialKS.210772 vs.120006. Both full verdicts remain FAIL.

**Next:** Keep A2 in the reference and preserve this negative in memory. Do not extend this failed parent, rerun its control or launch a seed/width sweep. Further calibration needs a supported new hypothesis or compatible saved positive evidence; no qualification or default adoption.

[Record and receipt provenance](records/readout-5830d399b6741ce7b3a2b992.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-5a2f259958a8784b19e78ff3

**Scope:** research_diagnostic; scientific; revision `3ca1ad7447f7d38dd2b4040f9f15829f7191d027bae364faba1f6008bc4a9b1d`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.06; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-5a2f259958a8784b19e78ff3.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2--abf642c42c5346ad096c29202e4716db535c393c113478552133c1c22761ddbd · readout-5bc30fee4d41b69294b1e32e

**Scope:** current; scientific; revision `7c08dcbf6b2e1a90368368ba328f3597a260381ca68b67f2ac28904050ce3ee1`.

Under the unchanged fixed toy-host movement/convergence gates, larger constant learning rates and paired R1/R2 gamma schedules may improve the failed Modern GAN training adaptation.

**Observed:** {'FAIL': 1, 'PASS': 3}; wall seconds 23.971; mechanism `floor_constant`.



Bounded R1/R2 Modern GAN toy-host configuration {'lr': 0.0085, 'reg_coeff': 1.0, 'reg_coeff_end': 0.1} under the fixed source/protocol/runtime produced [{'task': 'two_pole', 'gate_status': 'PASS', 'metrics': {'grad_med': 0.6198654174804688, 'mean_abs': 0.8125556111335754}}, {'task': 'unused_token_hold', 'gate_status': 'PASS', 'metrics': {'concept_move': 0.9999794960526742, 'unused_hold': 0.997301101218909}}, {'task': 'ae_gan_hold', 'gate_status': 'PASS', 'metrics': {'hold': 0.06419558823108673, 'recon_mse': 0.006500412710011005}}, {'task': 'trajectory', 'gate_status': 'FAIL', 'metrics': {'identity_mse': 0.23911647498607635}}]. Ordinary prerequisite stopping retained every unmeasured requirement.

**Next:** Stop this exact failed configuration; review the provisional convergence screen before another parameter study. No downstream diagnostic, seed repeat or default adoption follows.

[Evidence](../../reports/forge/technique-receipts/3c6a84341114445da149f04ab7541f01.json) · [Record](records/readout-5bc30fee4d41b69294b1e32e.json)

### recipe-prior-refactor-bcap-pure-v1 · readout-5bfc1b05b07fc9c2660b0b9d

**Scope:** research_diagnostic; scientific; revision `9d6985b51f1aecf05952b60df4f6a337b8d14a21adb3bc782e9298dd2a3784e1`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 164.774; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-5bfc1b05b07fc9c2660b0b9d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-prior-fresh-gaussian-v1 · readout-5ed4195bb58c87dc84a60827

**Scope:** research_diagnostic; scientific; revision `535d9c1c86a6b34cd606465d32a8071e842e37015d4641da4c7e90c281e7c78b`.

Replacing table codes with fresh N(0, I) latents of the same dimension loses sampled-latent passes that the learned BISPF default holds.

**Observed:** {'BLOCKED': 1, 'FAIL': 10, 'PASS': 5}; wall seconds 1433.143; mechanism `structural`.

vector_unequal_width: hq=0.84082, mass_tv=0.024658; vector_unequal_mass: hq=0.89111, mass_tv=0.092139

Fresh N(0,I) latent: 5 PASS / 10 FAIL on the 15 supported sampled-MoG tests (33 blocked by design); identical verdicts to the frozen table; loses ring16_acquisition (hq 0.53) and mode_hold (hq 0.50) vs learned. See reports/prior-ablation-2026-10-25/README.md.

**Next:** Stop; the learned table beats an ordinary Gaussian latent on this suite.

[Record and receipt provenance](records/readout-5ed4195bb58c87dc84a60827.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v7 · readout-5f131a6d58b1cd725e77839c

**Scope:** research_diagnostic; scientific; revision `c4e5fde402efe8870108225c1b196666b720902000d5f550cbd252ba93d2774e`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.743; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for empf-v7 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-5f131a6d58b1cd725e77839c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-613fc91cba03a8c327465202

**Scope:** current; scientific; revision `8a77a5bd4e1f8f1c74bb516bb749e37a3170c8a80ed84cf0924d59b0b0d04bb6`.

Ordinary re-qualification of the unchanged selected r1r2 recipe (recipe-prior-refactor-r1r2-v1) on current develop source 2e8a97a0aa84 under protocol seed 0 on NVIDIA RTX A6000 records current-source discriminator_stability and shared Tier 1 goal-view verdicts.

**Observed:** {'FAIL': 4, 'PASS': 10}; wall seconds 380.031; mechanism `structural`.



Completed discriminator_stability seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (FAIL 4, PASS 10). Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact.

**Next:** Retain source-bound readout and compiled Forge reports.

[Evidence](../../) · [Record](records/readout-613fc91cba03a8c327465202.json)

### ka2-global-repair-v1 · readout-628b01d1adffd11312aff9b1

**Scope:** current; scientific; revision `71a50b7b10fdc6307c92b781224e73625f670377ffeb4a31b9f3abe4b34be108`.

The exact repaired global recipe acquires the five-word inverse task and transfers without task-specific optimizer tuning across every eligible required discriminator-stability task under ordinary prerequisites.

**Observed:** {'FAIL': 1}; wall seconds 5.982; mechanism `structural`.



Ordinary global candidate failed two_pole movement: mean_abs=0.0363295227<0.3; qualified tier0. Other25taskcells, including word, are UNKNOWN.

**Next:** Initial complete candidate rejected. Its subsequent separately bounded global coupled-rate search is concluded; no automatic paid continuation or default adoption.

[Record and receipt provenance](records/readout-628b01d1adffd11312aff9b1.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v1 · readout-65e4f5d253025443fdee7ee4

**Scope:** research_diagnostic; scientific; revision `45a5684f2961affd93bfb3cdd35b79928763c60cfe6a3e913492dc1bc77007e4`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.257; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for empf-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-65e4f5d253025443fdee7ee4.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-pure-global-v1 · readout-6675da6b575648d03b2182e9

**Scope:** research_diagnostic; scientific; revision `418fb58baf87609c9d8a8e95d60af662e1fe025f891dac4ccf81de16515f9b7b`.

Disabling every declared BISPF shape/step gate (one pure global recipe) loses some of the selected recipe's passes.

**Observed:** {'BLOCKED': 1, 'FAIL': 18, 'INCOMPLETE': 3, 'PASS': 13}; wall seconds 2744.178; mechanism `structural`.



Shape/step gates off: 13 PASS / 18 FAIL of 49; loses ring16_acquisition (sustained window), mode_hold, ring_extension, img_stripes2 and the five-word smoke (wall clock). See reports/prior-ablation-2026-10-25/README.md.

**Next:** Stop; keep the declared gates in the default recipe and investigate which single gate carries ring16/mode_hold.

[Record and receipt provenance](records/readout-6675da6b575648d03b2182e9.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-669ead9699347bdcc998eb81

**Scope:** research_diagnostic; scientific; revision `6cbcc912ba5da1d679be9e00dbda19345720ca7c420454bf9fe61ed6e5ba28a2`.

Evaluate whether unchanged selected atlas recipe (atlas-particle-normal-v4) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 83.775; mechanism `structural`.



Selected atlas recipe passes interp_img_patch_shift (5/5 sustained terminal checks: interp_rmse=0.0250, interp_peak=0.9358, midpoint_novelty=0.5305) and meets every final-step bound on interp_sparse_arc (interp_hq=0.9365, output_knn_interp_hq=1.000, 30/32 bins) and interp_saddle_barycentric (barycenter_surface_mae=0.0552, barycenter_hq=0.8701, 16/16 xy bins), but fails 5-check sustained live-iterate gates on the two vector tasks (3/5 passing terminal checks each).

**Next:** Test EMA/served weight evaluation or explicit local latent Jacobian smoothness regularization to stabilize the 5-check live suffix on sparse_arc and saddle_barycentric.

[Record and receipt provenance](records/readout-669ead9699347bdcc998eb81.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-69f698e7c003c96d01d05c05

**Scope:** research_diagnostic; scientific; revision `e60e24fa66aaac1d4eaba6608c568ed099ae4e65acf48830f3357eece151d7cd`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.198; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-69f698e7c003c96d01d05c05.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-6a7e566f412b2f89b3c4542e

**Scope:** research_diagnostic; scientific; revision `bd13bfd00479408abe45e88a8d3f986b3d18bff8859fb643e12ff9a41d6efd17`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.979; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-6a7e566f412b2f89b3c4542e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-6b39d05e0c481a63f7aa8aa3

**Scope:** research_diagnostic; scientific; revision `5407c9e4e87d23fc8f258d5a0bd0a1922666b1018ee39e61039efd6ed5c0e526`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.883; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-6b39d05e0c481a63f7aa8aa3.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v6 · readout-6bf4d676cc7a12b04a631fd2

**Scope:** research_diagnostic; scientific; revision `74aa6308874a9c6492244418f9a58b0768defe2de5ca0155a583d2edd63d8ae2`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 68.045; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for rhbi-v6 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-6bf4d676cc7a12b04a631fd2.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-no-output-noise-diagnostic · readout-6c0c2c5c6610ecfe73fe315d

**Scope:** current; scientific; revision `7bb7527f2fa60cf20120b07fae977f953e66716f31a40ba180240c35a376d803`.

Removing additive training output noise may prevent the clean learned-MoG native distribution from contracting late. This is a one-factor diagnostic of the training/scoring-law mismatch, not a predicted positive or a full-reference claim.

**Observed:** {'FAIL': 1}; wall seconds 5.876; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 0.13080213963985443, 'mean_abs': 0.2641167640686035}. One completed attempt cost 5.876473600 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/813505867f5348bfbe113791d78ce0c5.json) · [Record](records/readout-6c0c2c5c6610ecfe73fe315d.json)

### bispf-ddgan-energy-unconstrained-v3 · readout-6d9c721048b23e3d6b55e971

**Scope:** research_diagnostic; scientific; revision `6f3641b574a34f3ac199275ce9a2d9a44cc5fb77f281e583777be601c73221b4`.

Removing per-objective direction blending from the same low-mobility summed GAN/energy objective repairs low-noise conditional calibration while preserving the Gaussian marginal stability and genuine adversarial game.

**Observed:** {'FAIL': 2}; wall seconds 739.185; mechanism `structural`.



Both complete DDGAN tasks FAIL after the final registered one-field repair. Seven full-budget failures and two resource-incomplete attempts conclude the three-round campaign; no global recipe passes either strict task.

**Next:** Stop all exact revisions with zero ordinary qualification. Retain generation-only subgate gains and complete failures. A future explicitly scoped timestep-conditioning/capacity architecture cohort needs matched controls and unchanged scientific gates; current results do not prove a capacity limit or authorize another seed/rate trial.

[Record and receipt provenance](records/readout-6d9c721048b23e3d6b55e971.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-6efd2f8cd626467d353678be

**Scope:** research_diagnostic; scientific; revision `14e65222b83d34897962a7c21fc26cf2870000df821adbdd4f2a0de3956f13a4`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.048; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-6efd2f8cd626467d353678be.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-7018a47a2fdd016bbe6b2658

**Scope:** research_diagnostic; scientific; revision `37c1bfc115abcc2a9a529ffb799cd1af5c1dd03601b7f1b077bed51d914969fa`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.342; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-7018a47a2fdd016bbe6b2658.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-704dd415ee7878a1bbfbe7d9

**Scope:** research_diagnostic; scientific; revision `df997534e85ad8b5331ea151e33ca1f306141564fad67451491dd7a821220815`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.911; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-704dd415ee7878a1bbfbe7d9.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-v2 · readout-7450f3d30d639bad8e4f25e5

**Scope:** research_diagnostic; scientific; revision `e558bd3a50a3746df01baaf4759aaae09b48f465012d157796b76602cd2ce2e0`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 170.099; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/bispf-dcgan_performance_b64-8980e3805d46c8aa.json) · [Record](records/readout-7450f3d30d639bad8e4f25e5.json)

### empf-v2 · readout-749aa0f20c897f4e53d4d97c

**Scope:** research_diagnostic; scientific; revision `fc09c6b1f8ae77167ca31db0a4d9ef252dc9d6b92aed98e3d4b53a168ff0ac15`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.089; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for empf-v2 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-749aa0f20c897f4e53d4d97c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v4 · readout-75cdbfb23dff3c1136ae7422

**Scope:** current; scientific; revision `b44c5b2df075ea9eb81d850ce3dba4c04f3b6b8f3aff87963708d729ce03f2c8`.

Restoring fixed K = min(max_prototypes, B) prototype banks for B < 100 and dim != 1 Born-Infeld settling sharpness while normalizing innovation shock by 2 * sum(Var(X)) and fusing penalty autograd/scalar syncs restores model_vae_categorical Tier 1 suffix stability and throughput, allowing empf-v4 to clear Tier 1 cleanly and realize empf-v3's 1D quantile-W2 floor-subtracted cooling (gaussian1d_stability), Poisson empty-cell shift acceleration (target_shift_recovery), and prototype-referenced deep harmonic-capture lock hysteresis (ring_hold, ring_extension) under one global clock-free constant-LR DualNorm recipe.

**Observed:** {'FAIL': 5, 'INCOMPLETE': 1, 'PASS': 9}; wall seconds 1391.266; mechanism `structural`.

model_ae_reconstruction: modes=4, precision=0.26587, mass_tv=0.049805; ae_gan_hold: recon_mse=0.0065832

On ordinary discriminator_stability (Rev 16) under protocol seed 0 on NVIDIA RTX A6000, empf-v4 passed model_vae_categorical and model_vae_hard after removing small-batch online prototype growth, but stopped at Tier 1 (8/11 required Tier 1 PASS; ring16_acquisition and model_ae_gan_multimodal FAIL; precision_gaussian_fp32 and precision_gaussian_mxfp8 FAIL; five_word_joint_smoke INCOMPLETE at 900.3s wall-time limit).

**Next:** Keep fixed K=min(max_prototypes,B) small-batch banks, variance-normalized innovation shock, fused autograd syncs, and zero-overhead discrete-simplex scaling while restoring empf-v3's thermal_decay=0.92 and cool-gated Born-Infeld batch_gain in empf-v5.

[Record and receipt provenance](records/readout-75cdbfb23dff3c1136ae7422.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-no-critic-penalty · readout-75d5cdc5a88167d59ceff1b2

**Scope:** calibration_diagnostic; scientific; revision `5bbe11c14ddca9d5334589d6901e38838d627fdecee94d6bd6aa86e1f1e76125`.

Removing the public critic gradient penalty tests whether cheap stability gates reject a substantively weakened formulation before quality work.

**Observed:** {'FAIL': 1}; wall seconds 21.672; mechanism `structural`.

mode_hold: modes=0, hq=0

Registered mode-hold diagnostic FAIL: 0/8 modes, HQ 0, zero passing observations in 24 checks; 1,200 complete updates and 21.671807714 paid seconds. No infrastructure error. Independent reference outcomes remain unknown.

**Next:** Conclude this mode-only diagnostic. Preserve the negative in memory; do not fill this exact calibration profile solely for adoption, since all three smoke predicates fail and the unchanged criteria require a positive with no false rejection. A new justified profile or independently compatible positive reference is needed; no general downstream or universal algorithm claim.

[Record and receipt provenance](records/readout-75d5cdc5a88167d59ceff1b2.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-7650981f9a48a164aebbd631

**Scope:** calibration_diagnostic; scientific; revision `2cde93ebff750cc551fcfdae18d82b4edfc60f9d0aca49713968faa5823d3591`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'PASS': 1}; wall seconds 11.199; mechanism `structural`.

img_intensity2_residual16: modes=2, hq=0.90625

The explicit published residual16 intensity host passed all sustained/terminal requirements using the current shared K3P formulation, fixed named initialization and clean live scoring.

**Next:** Bind published vector/native architecture and initialization policies explicitly through shared public components before expanding the full reference matrix. Keep this positive, prior failures and costs in separate compatible cohorts. Register any further diagnostic narrowly; no automatic control/reference sweep or adoption is justified by one image pass.

[Record and receipt provenance](records/readout-7650981f9a48a164aebbd631.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-7781bffb3ec011c9b933c0b5

**Scope:** research_diagnostic; scientific; revision `983c930b0b5924f062e918302e8b9c6af6605ef17d190f9b0518c75f1a0b8e4f`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 167.125; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-7781bffb3ec011c9b933c0b5.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v4 · readout-78930a17ca5f4c4926f626a0

**Scope:** current; scientific; revision `a555da62030ea9cce3aa3d9a76e1f2e8ab3ffe86f4347f9b865cf52fafc7c459`.

Adding AKT empirical OT 1D horizon scaling (rhbi_akt_horizon=true), slack-gain critic boost (rhbi_slack_gain=0.35), deterministic Weyl-phased balanced-Sinkhorn geodesic chord regularization with zero-mean critic gauge (rhbi_chord_weight=0.25, ebi_beta=12.0, rhbi_gamma_gauge=0.03), and an impedance-matched 5-channel Kantorovich-Hellinger thermal controller (rhbi_thermal_tau=0.084) to rhbi-v3 preserves all 27/27 Tier 1 requirements and ring16_acquisition while passing mode_hold and vector_overlap under one global clock-free, shape-gate-free recipe on protocol seed 0.

**Observed:** {'FAIL': 14, 'PASS': 24}; wall seconds 3425.912; mechanism `structural`.

staggered100: precision=0.3086, mass_tv=0.11332; vector_anisotropic: hq=0.91748, mass_tv=0.14876

prediction_observed

**Next:** review_saved_diagnostics

[Record and receipt provenance](records/readout-78930a17ca5f4c4926f626a0.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-7893dd108a213b15eb871284

**Scope:** research_diagnostic; scientific; revision `dacdf89fb69d40f25387d58502405ca53495698392c7f7c27334210496660f3c`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.714; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-7893dd108a213b15eb871284.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v1 · readout-78b2f38e0f8c24f678b47fa2

**Scope:** research_diagnostic; scientific; revision `119e0a54eeb8727fbc823acf61cf2cfa364a63a69bed6b46d1cefcbb8f49d1cf`.

Swapping MoGParticlePrior for a technique-owned unstandardized ParticlePrior plus NormalNoiseManipulator (sigma=0.005) unblocks public_trainer execution across the 48-task full suite while keeping the public atlas preset recipe fixed under protocol seed 0.

**Observed:** {'BLOCKED': 1, 'FAIL': 12, 'PASS': 6}; wall seconds 3631.847; mechanism `structural`.

vector_unequal_mass: hq=0.99146, mass_tv=0.014307; vector_anisotropic: hq=0.97925, mass_tv=0.005778

Full-suite diagnostic (prior-ablation-full-suite-v1, seed 0, deterministic_orthogonal): technique-owned prior_mechanism (unstandardized ParticlePrior + NormalNoiseManipulator sigma=0.005, deterministic sigma=0.0 on enumerated image tasks) with unchanged atlas preset recipe unblocked all 25 public_trainer tasks (19 executed, 6 continuations gated by parent clean-variance verdicts, 23 non-policy/unrouted hosts blocked). Both study signatures confirmed: grid100 precision=0.99737 (>=0.95, 100/100 modes, mass_tv=0.02576, center_rms_sigma=0.13426; rotated100 precision=0.9993, staggered100 precision=0.9969) and gaussian1d_smoke mean_error_sigma=0.0052 (<=0.20 PASS). Total 6 PASS (gaussian1d_smoke, vector_two_broad, vector_anisotropic mass_tv=0.0058, vector_overlap, vector_spiral, img_stripes2 hq=1.0000), 12 FAIL, 1 runtime BLOCKED (zero_tanh_smoke), 29 preflight/dependency not run.

**Next:** Test a clean-training Atlas study (output_noise_mode='fixed', output_noise_std=0.0 paired with prior_mechanism NormalNoiseManipulator sigma=0.005 and min(birth_death_cells, num_particles // 2) cell cap) so the generator carries full within-mode variance under clean evaluation without altering row-local particle transport.

[Record and receipt provenance](records/readout-78b2f38e0f8c24f678b47fa2.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-7a4a425e8499eff696ed464b

**Scope:** research_diagnostic; scientific; revision `ae737603cb7bc7d8e9f7e32ee3775dacd20997e2c06d99648ebade1665211705`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.974; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-7a4a425e8499eff696ed464b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-throughput-v1 · readout-7a4fda449cb2e080c1aebf91

**Scope:** research_diagnostic; scientific; revision `5519ffc77cf0cfd46928ca25fbbf29efb9a642d8da333482419f4c9af3a4300d`.

Preserve both incumbent width64 image passes under the unchanged full numerical gates, using one declared throughput recipe. Performance does not imply quality.

**Observed:** {'PASS': 2}; wall seconds 35.621; mechanism `structural`.

img_bars4_w64_optimizer_diagnostic: modes=4, hq=1; img_stripes2_w64_optimizer_diagnostic: modes=2, hq=1

Both width 64 image tasks PASS all original sustained gates at 600 updates: HQ 1,2 stripe modes and 4 bar modes, with passing suffixes 12 and 9.

**Next:** Retain this positive diagnostic beside ordinary quality and the completed performance matrix; no natural-image quality or automatic promotion claim.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-7a4fda449cb2e080c1aebf91.json)

### bispf-optimized-quintic-v1 · readout-7b017c8ed900b677ebf6bf55

**Scope:** research_diagnostic; scientific; revision `97cff463de780cdb270a5cb968799379fe2eef3bbad671fd59df0c3d9515eeb0`.

Re-qualification: the unchanged selected BISPF recipe on develop (after the review RNG-contract and declared-gate fixes) reproduces its recorded full-suite outcomes.

**Observed:** {'BLOCKED': 1, 'FAIL': 16, 'INCOMPLETE': 2, 'PASS': 20}; wall seconds 3203.035; mechanism `structural`.

grid100: precision=0.30497, center_rms_sigma=1.4971, mass_tv=0.17028, radial_ks=0.42394

Re-qualification on develop source b4d7f2c42: all 29 recorded PASS/FAIL verdicts reproduced (18 PASS, 11 FAIL); new: clockfree_audit PASS, target_shift_recovery and four API model toys FAIL. 19 PASS / 16 FAIL / 7 blocked / 5 not run / 2 incomplete of 49. See reports/prior-ablation-2026-10-25/README.md.

**Next:** Keep as the default recipe; register a structural idea that combines learned precision with frozen-table mass evenness on the native grids.

[Record and receipt provenance](records/readout-7b017c8ed900b677ebf6bf55.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-7b4757c5a675b179621bffff

**Scope:** research_diagnostic; scientific; revision `53feece7ca4ae9abbd684b86f7cabf243277985e33c1cd6eaceb1a0881014973`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.382; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-7b4757c5a675b179621bffff.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-7bbb55d0ca2d3c3b99f56d10

**Scope:** research_diagnostic; scientific; revision `126a196fb9b0bca2fb4836c8fd8a23ab02ef28cf24e6068342f97302a2061ec3`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.095; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-7bbb55d0ca2d3c3b99f56d10.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v1 · readout-7ede56c9affbd90cbf1159f2

**Scope:** research_diagnostic; scientific; revision `5818d9b925e94cd23f9c7e68ad425cef0b26d5fa993d6ece28f0f7bb3a954810`.

Swapping MoGParticlePrior for a technique-owned unstandardized ParticlePrior plus NormalNoiseManipulator (sigma=0.005) unblocks public_trainer execution across the 48-task full suite while keeping the public atlas preset recipe fixed under protocol seed 0.

**Observed:** {'FAIL': 10, 'PASS': 11}; wall seconds 2876.824; mechanism `structural`.

img_blobs4_residual16: modes=4, hq=1; img_intensity2: modes=2, hq=0.96875

Atlas particle-normal prior replication study executed 21 of 48 tasks in research_diagnostic scope under seed 0: 11 passed, 10 failed, 27 tasks unexecuted due to preflight blockers. Prediction on grid100 observed (precision >= 0.95); falsifier avoided (mean_error_sigma <= 0.20). Overall study outcome is incomplete; qualified_tier 0 grants no production qualification or merge authorization.

**Next:** Resolve behavior adapter evaluator source hash mismatches in preflight checkers, execute unblocked tasks to complete study scope, and address intra-mode covariance scaling.

[Record and receipt provenance](records/readout-7ede56c9affbd90cbf1159f2.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-811350d1fcece799e90ae98f

**Scope:** research_diagnostic; scientific; revision `9364f776dfd64e91fe22f37985fb4d729ad3106c5d8139234ee5e5670ad4306c`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.626; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-811350d1fcece799e90ae98f.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p--9f4bc2973d95cf1545e009ec044ed582a318e6d869d62f4d99f6eee529d1d93d · readout-813a2ad71ca3580b4a8d46c1

**Scope:** current; scientific; revision `f415e58283e06eaf2679a31dbf0317b69bda8fd3e1a880e62209ac9318f1eaf8`.

Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.

**Observed:** {'FAIL': 1}; wall seconds 5.976; mechanism `floor_constant`.



This exact global configuration failed the first two_pole movement gate. Tier1 is0/5; word and25remainingcells are UNKNOWN. No configured standard or default adoption.

**Next:** Bounded global search concluded. Retain the exact historical family incumbent as unqualified evidence. No further paid stage is authorized; require a separate bounded global proposal before any continuation.

[Record and receipt provenance](records/readout-813a2ad71ca3580b4a8d46c1.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-misgan-observed-v1 · readout-8287ac80f671694274d99b8a

**Scope:** research_diagnostic; scientific; revision `5a3af62f0de5000d2215894277efb258ab83653b8621a86dacaadb2aef11624c`.

Retain BISPF core and distinguish incomplete-data recovery from learned multimodal conditional amortization with the frozen scorer and controls.

**Observed:** {'FAIL': 1}; wall seconds 621.916; mechanism `structural`.

model_misgan_four_mode: modes=4, hq=0.99854, mass_tv=0.16968

Full scoped four-mode run completed1200updates/24checks and FAILed all5terminal complete gates. Endpoint generatorHQ=0.998535, massTV=0.169678, ambiguousHQ=0.986084, branchTV=0.087891. Preserve every terminal failed bound; no ordinary or historical100-mode pass credit.

**Next:** Stop this exact revision. The sole bounded composite masked-quantile transport plus cosine-settling followup changes the failed mass/settling mechanisms while keeping the frozen task, initializer, prior, sampling and1200update protocol.

[Record and receipt provenance](records/readout-8287ac80f671694274d99b8a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-828f21ff6556847d8f000678

**Scope:** research_diagnostic; scientific; revision `79e308b48506dccbf888516575643268a7499751b98b7d9802fbf7c56b2ad186`.

Guarding LossSurprise.decide against critic-only SettledReopenGuard witnesses (witness_roles == {'critic'}, where the critic ladder has stepped down while the generator ladder remains unsettled at full rate s=1.0) prevents premature step-35 critic re-empowerment on N=12 mode_hold, recovering 8/8 mode_hold and unblocking ring_hold, ring_extension, and target_shift_recovery while retaining birth_death_cells=256 and NormalNoiseManipulator(sigma=0.0065) to preserve all 6/6 vector, 4/4 image, 2/2 VAE, and 2/2 clockfree passes for >= 24 PASS / 0 INCOMPLETE with zero task regressions relative to both atlas-particle-normal-v2 and atlas-particle-normal-v3 under protocol seed 0.

**Observed:** {'FAIL': 6, 'INCOMPLETE': 1, 'PASS': 8}; wall seconds 471.024; mechanism `structural`.

ae_gan_hold: recon_mse=0.0099752

Cancelled partial diagnostic run of atlas-particle-normal-v4 on pre-fix commit 6c895b5924d0 (clockfree_audit INCOMPLETE before public_components/zero_tanh_smoke fix in 6261a5458c1f).

**Next:** Mark revision 79e308b48506 superseded by efebaebc63fb and retain clean reverified v4 evidence.

[Record and receipt provenance](records/readout-828f21ff6556847d8f000678.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-85674fad5d145e6171cc44af

**Scope:** research_diagnostic; scientific; revision `4f4f510992ab2ef6f5c12bb7ce8cf5ee71218cdf63e936cee281413007ae01e2`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.079; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-85674fad5d145e6171cc44af.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-frobenius-v1 · readout-85a714c5fc9c0b038c400b22

**Scope:** research_diagnostic; scientific; revision `c6dadb65374f5c93fe3922c2e10e32b5964105d4dc13ff80b97773c85b9f1aff`.

The global frobenius rule retains the native BISPF pass set on both unchanged width-64 image tasks, including a positive stripes acquisition pass, under the same seed, public initializer, data sequence, prior, sampling, 600 updates and 24 checks.

**Observed:** {'FAIL': 2}; wall seconds 29.708; mechanism `structural`.

img_stripes2_w64_optimizer_diagnostic: modes=1, hq=0.96875; img_bars4_w64_optimizer_diagnostic: modes=2, hq=0.8125

Frobenius convolution rule fails both width64 image gates at600 updates: stripes HQ0.96875 but1 mode versus2 required, bars HQ0.8125 with2 modes versus4 required. Both terminal passing suffixes are0. Public API runs remain finite with no unintended RNG deviations.

**Next:** Stop this exact Frobenius revision; do not run full performance or ordinary qualification for selection. Preserve the collapse failure and its actual-training GIFs; no outcome-based rerun, seed repeat or gate change.

[Record and receipt provenance](records/readout-85a714c5fc9c0b038c400b22.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### forge-onboarding-anchor-ablation · readout-8aacfe961f0e62b062e621fa

**Scope:** calibration_diagnostic; scientific; revision `b10c31a0de40294be8966579fbe56e84e06e24dea1e3576e65b35148f610d350`.

Removing the public critic EMA-anchor term may weaken short-horizon adversarial movement; the fixed CPU smoke gate should reject a harmful ablation before larger tests.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 16.383; mechanism `structural`.

ae_gan_hold: recon_mse=0.004509

All three registered smoke cells completed: two_pole FAIL (mean_abs0.1048249<0.30; grad_med0.0767846<=1), unused_token_hold PASS, learned-MoG ae_gan_hold PASS. The entire smoke decision is FAIL; no quality-reference outcome is inferred.

**Next:** Retain all nine cheap-cell outcomes and costs. Keep ordinary qualification stopped. Review the exact historical/public-host parity differences before deeper reference spending; reserve the required GPU window and register only justified missing reference cells. Do not tune thresholds or treat unmeasured reference labels as failures.

[Record and receipt provenance](records/readout-8aacfe961f0e62b062e621fa.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bcap-dualnorm-convolution-v1 · readout-8b207a728977752808cf002c

**Scope:** research_diagnostic; scientific; revision `dd0702302681db384f1f1a23e10bfd29a40bae4ccbfea5e01e872d8a4d1a9061`.

Per-offset smoothed DualNorm with explicit grouped convolution and transposed convolution layout can train the four previously unsupported image hosts at the selected fixed rates.

**Observed:** {'FAIL': 4}; wall seconds 100.267; mechanism `structural`.

img_intensity2: modes=2, hq=0.96875; img_stripes2: modes=2, hq=1; img_bars4: modes=2, hq=0.65625

Per-offset smoothed DualNorm completed all four unchanged image tasks: {'FAIL': 4}; all 2400 declared updates ran. The prior four optimizer setup errors remain under the original source. Constant rates, fixed smoothing, seed0, no retries or annealing.

**Next:** Use the recorded quality/coverage trajectories to assess the convolution adaptation. Keep the six original Tier1 passes and full original Tier2 readout; other eleven failures and Tier3 remain unaddressed.

[Evidence](../../reports/forge/bcap-convolution/readout.json) · [Record](records/readout-8b207a728977752808cf002c.json)

### forge-onboarding-anchor-ablation · readout-8bfdfb8d2b1c3daf55f9b0bc

**Scope:** current; scientific; revision `e356a8bd7091ac1e130819a62c362fe3b40a5119a4c457793faa889abd9785c7`.

Removing the public critic EMA-anchor term may weaken short-horizon adversarial movement; the fixed CPU smoke gate should reject a harmful ablation before larger tests.

**Observed:** {'FAIL': 1}; wall seconds 5.449; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 0.07678461074829102, 'mean_abs': 0.10482493042945862}. One completed attempt cost 5.449001096 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/299aba837ed24be8a61b0b48074fc7a1.json) · [Record](records/readout-8bfdfb8d2b1c3daf55f9b0bc.json)

### rhbi-v5 · readout-8d88061b544c2ad1873e7160

**Scope:** current; scientific; revision `9d01bd567cb5aade5f0968102173a4f1d15002619f8afdc59a751591ad5c13ee`.

Upgrading RHBI from rhbi-v4 to rhbi-v5 with Poisson-Weibull 1D AKT empirical OT horizon normalization (rhbi_akt_mode='weibull'), false-shock-proof thermal reopening (rhbi_thermal_shock_mu=0.35), state-driven settling Born-Infeld Lipschitz sharpness (rhbi_settle_beta=22.0), and row-local calibrated-relativistic conditional fibre critic blending (rhbi_fibre_rel_blend=0.7) preserves all 27/27 Tier 1 requirements and Tier 2 counted passes while tightening conditional residual_student alignment and passing quantized 1D NVFP4 acquisition under one global clock-free, shape-gate-free recipe on protocol seed 0.

**Observed:** {'FAIL': 5, 'PASS': 10}; wall seconds 1152.914; mechanism `structural`.

five_word_joint_smoke: modes=5, mass_tv=0.018945

Completed discriminator_stability Tier-1 seed-0 CUDA execution for rhbi-v5 (10 PASS, 5 FAIL). Weibull 1D AKT horizon normalization unlocked precision_gaussian_nvfp4 (PASS, cdf_ks=0.02344) and preserved precision_gaussian_mxfp8 (PASS, cdf_ks=0.03628), but rhbi_settle_beta=22.0 activated prematurely at residual < 0.22 (above thermal_tau=0.084), over-stiffening ebi_beta during active mode acquisition on ring16_acquisition, ae_gan_hold, model_ae_gan_multimodal, and model_vae_categorical.

**Next:** Preserve rhbi-v5 readout as faithful negative/partial evidence and advance to rhbi-v6 gating settling Lipschitz sharpness strictly inside the thermal equilibrium regime.

[Record and receipt provenance](records/readout-8d88061b544c2ad1873e7160.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-8e9d62a78cd58a9fc3bde49e

**Scope:** research_diagnostic; scientific; revision `a3b5c6b0de7f055ac08d345d220442396c3cd6e375cd07cd58c968738242e3ad`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 63.717; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-8e9d62a78cd58a9fc3bde49e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-8fb398d9386aae96aa377eea

**Scope:** research_diagnostic; scientific; revision `23859bfe19bf5e6b58640f2aab6cabf8f9b0c1c19a39fbbf7870490f36b7707a`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.681; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-8fb398d9386aae96aa377eea.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v11 · readout-90b0065d5a734ce320912d8b

**Scope:** research_diagnostic; scientific; revision `05f71faf03f06323d92bc7def373a5310d32c91bea38bf62b760725e29415f55`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 69.08; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for rhbi-v11 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-90b0065d5a734ce320912d8b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-946a921be2625c9d807c6d1f

**Scope:** research_diagnostic; scientific; revision `8c1d7b083639d1f12c8d5398f27d03bb9e437b991f9e56178eaab2914d1e4fcc`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.482; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-946a921be2625c9d807c6d1f.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-bcap-matched-v1 · readout-95e0cdfdc2f043a0046285c1

**Scope:** calibration_diagnostic; scientific; revision `5db772f81cafe59b642b5707e600380fb49c3d42623b5c769e03dd677bbbfafd`.

Replace only the critic penalty by frozen historical bcap while retaining the K3P optimizer, noise, A2, host and sampling contracts.

**Observed:** {'FAIL': 1}; wall seconds 86.198; mechanism `structural`.

grid100_affine_paired_laws_v1: precision=0.99317, center_rms_sigma=0.05434, mass_tv=0.032, radial_ks=0.13169

Preregistered full7000-update grid100_affine_paired_laws_v1 FAIL; sustained coverage PASS, 0/5 terminal joint passes. Independent100k clean/live holdout: centre RMS 0.054340458sigma, covariance bias -0.285121929, radialKS 0.131685162, massTV 0.032000000, precision 0.99317. All G/D/prior updates complete, finite/hooks checks PASS, unintended RNG deviations0, independent receipt/artifact/evaluator audit PASS. Unique paid 86.197886095993s. No ordinary qualification or promotion; all unknown cells preserved.

**Next:** Stop these exact revisions and this selected registration. Before another GPU study, perform a bounded zero-training saved-state audit of critic radial and centering gradients against per-mode moments. Only a supported substantive hypothesis warrants a new frozen registration. Preserve A2 and sigma .025; no coefficient, cap, width, amplitude or seed sweep, best-checkpoint/EMA substitution, 14k continuation or automatic matrix expansion.

[Record and receipt provenance](records/readout-95e0cdfdc2f043a0046285c1.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas · readout-99058c02555c6e8edcc6e461

**Scope:** research_diagnostic; scientific; revision `7daf8b4faa3fe09ee7f13bd43f687f9daacf2424f8de990c4eeedec09396b224`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 69.454; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/atlas-dcgan_performance_b64-84b1001259e8b126.json) · [Record](records/readout-99058c02555c6e8edcc6e461.json)

### recipe-prior-refactor-release07-gan-v3-v1 · readout-99d51a04a1d1358bfd32303e

**Scope:** research_diagnostic; scientific; revision `3e28f9f94731bb06aa9e11a7826cef20a04758951cc283ef6476e988af605e52`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.726; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-99d51a04a1d1358bfd32303e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-9ba1e04d7902d860f201b335

**Scope:** research_diagnostic; scientific; revision `dd75ccf13c045d644d551963d6867b3d0c57b112ffd14dc988c4ac85624df3e5`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.736; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-9ba1e04d7902d860f201b335.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-9c29229a2baba56de3b40845

**Scope:** calibration_diagnostic; scientific; revision `989a12b2dd014078e0e5dcf7c674be243b5a73bee19289c9500964a941369c69`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 3}; wall seconds 38.915; mechanism `structural`.

img_bars4: modes=2, hq=0.9375; mode_hold: modes=5, hq=0.99878; img_intensity2: modes=0, hq=0

The registered merged-develop quick screen completed 0/3 PASS on GPU 0: mode_hold 5/8 modes with hq=0.998779296875; img_bars4 2/4 modes with hq=0.9375; img_intensity2 0/2 modes with hq=0. All 24 observations per task were recorded, with zero passing suffixes and zero unintended RNG deviations. Three scientific FAILs, no execution errors; paid wall time 38.915211247018306 seconds. Live scoring used clean public prior samples; mode_hold used learned MoG and the image hosts used explicit cloud enumeration.

**Next:** Keep the full reference denominator and stop automatic expansion to ablations or the full reference matrix. Review host/formulation solvability and initial-state parity before selecting another bounded scientific diagnosis. Correct future sampling declarations and CUDA telemetry without rewriting these receipts or rerunning failed science just for reporting. The separately registered physical two-GPU pilot still needs a joint device window; this serial screen does not satisfy it.

[Record and receipt provenance](records/readout-9c29229a2baba56de3b40845.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### release07-gan-v3-cloud-v1 · readout-9c48b366ebe60b61721c269d

**Scope:** calibration_diagnostic; scientific; revision `3f2629ea8a1a6c9176d021ccdb99df48c2f7f67296946110288d59f3015ae14d`.

Evaluate the released v0.7 GAN v3 effective recipe on its explicit z4/batch256/unstandardized cloud host with named caller initialization.

**Observed:** {'FAIL': 1}; wall seconds 72.155; mechanism `structural`.

grid100_release07_cloud_named_v1: precision=0.76844, center_rms_sigma=0.57414, mass_tv=0.15151, radial_ks=0.10595

Preregistered full7000-update grid100_release07_cloud_named_v1 FAIL; sustained coverage FAIL, 0/5 terminal joint passes. Independent100k clean/live holdout: centre RMS 0.574141532sigma, covariance bias -0.266083317, radialKS 0.105947048, massTV 0.151510000, precision 0.76844. All G/D/prior updates complete, finite/hooks checks PASS, unintended RNG deviations0, independent receipt/artifact/evaluator audit PASS. Unique paid 72.154768106993s. No ordinary qualification or promotion; all unknown cells preserved.

**Next:** Stop these exact revisions and this selected registration. Before another GPU study, perform a bounded zero-training saved-state audit of critic radial and centering gradients against per-mode moments. Only a supported substantive hypothesis warrants a new frozen registration. Preserve A2 and sigma .025; no coefficient, cap, width, amplitude or seed sweep, best-checkpoint/EMA substitution, 14k continuation or automatic matrix expansion.

[Record and receipt provenance](records/readout-9c48b366ebe60b61721c269d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2-global-repair-v1 · readout-9e297f3a07ea06f0abc19272

**Scope:** current; scientific; revision `5a20a41ac1b78983611d3cfd765d572c436f93fa5b00ace5dba9934c288e4107`.

The exact repaired global recipe acquires the five-word inverse task and transfers without task-specific optimizer tuning across every eligible required discriminator-stability task under ordinary prerequisites.

**Observed:** {'FAIL': 1}; wall seconds 6.028; mechanism `floor_constant`.



Ordinary global candidate failed two_pole movement: mean_abs=0.0394435711<0.3; qualified tier0. Other25taskcells, including word, are UNKNOWN.

**Next:** Initial complete candidate rejected. Its subsequent separately bounded global coupled-rate search is concluded; no automatic paid continuation or default adoption.

[Record and receipt provenance](records/readout-9e297f3a07ea06f0abc19272.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v1 · readout-9e900493324c626b8c842923

**Scope:** research_diagnostic; scientific; revision `d21b0e556ee2739eea4c59d9cfe23b31a699161ccc7e12622a6ed75bf6cf38f5`.

Swapping MoGParticlePrior for a technique-owned unstandardized ParticlePrior plus NormalNoiseManipulator (sigma=0.005) unblocks public_trainer execution across the 48-task full suite while keeping the public atlas preset recipe fixed under protocol seed 0.

**Observed:** {'FAIL': 11, 'INCOMPLETE': 1, 'PASS': 10}; wall seconds 4658.665; mechanism `structural`.

ring_hold: modes=8; ring_extension: modes=8; vector_unequal_mass: hq=0.97876, mass_tv=0.03252

Atlas particle-normal prior replication study executed in research_diagnostic scope under seed 0: prediction on grid100 observed (precision >= 0.95), falsifier avoided (mean_error_sigma <= 0.20). Study outcome is incomplete with qualified_tier 0; diagnostic findings grant no production qualification or merge authorization.

**Next:** Review saved diagnostics and address intra-mode covariance scaling under clean evaluation.

[Record and receipt provenance](records/readout-9e900493324c626b8c842923.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v1 · readout-9f0d62f87d1ff351cfef7f98

**Scope:** research_diagnostic; scientific; revision `c7e4c12988d060ec0bb8d903651ffbae3d2537e4d5b360a3bfa75a6e860de9fb`.

Swapping MoGParticlePrior for a technique-owned unstandardized ParticlePrior plus NormalNoiseManipulator (sigma=0.005) unblocks public_trainer execution across the 48-task full suite while keeping the public atlas preset recipe fixed under protocol seed 0.

**Observed:** {'FAIL': 14, 'INCOMPLETE': 3, 'PASS': 17}; wall seconds 3036.229; mechanism `structural`.



Atlas particle-normal prior replication study executed in research_diagnostic scope under seed 0: prediction on grid100 observed (precision >= 0.95), falsifier avoided (mean_error_sigma <= 0.20), unblocking transfer behavior and model toys.

**Next:** Review saved diagnostics and address intra-mode covariance calibration under clean evaluation.

[Record and receipt provenance](records/readout-9f0d62f87d1ff351cfef7f98.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-a00bfca2993b57da0e7ca161

**Scope:** research_diagnostic; scientific; revision `a298bfb2c0afa14c7c96dd05071ee3c54d5f6c7b7ca05fac221407420948c0e7`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 61.483; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-a00bfca2993b57da0e7ca161.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v7 · readout-a211bf674ae6989087e57402

**Scope:** research_diagnostic; scientific; revision `bf914aab956af397d25e528718d105f1cad770c8055474ed213e3257d55ddcc4`.

Evaluate whether unchanged selected empf recipe (empf-v7) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 3}; wall seconds 88.381; mechanism `structural`.



Selected empf-v7 recipe achieves final-step interp_hq=0.8254 (output_knn_interp_hq=0.8421, 27/32 bins) on interp_sparse_arc via DualNorm polar geometry, but fails endpoint_hq (0.6667 < 0.85) and fails img_patch_shift (zero-output collapse) and saddle_barycentric (barycenter_hq=0.4167) for 0/3 sustained PASS.

**Next:** Diagnose zero-sigma 16-particle image collapse under DualNorm hybrid polar updates before tuning latent smoothness.

[Record and receipt provenance](records/readout-a211bf674ae6989087e57402.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-a2a29effc3b7a25d04c3f822

**Scope:** research_diagnostic; scientific; revision `3a9716bd0e5d7af47d0efe8cd2606257823688e0dda24e5fa895747edc42efb9`.

Evaluate whether unchanged selected ka2 recipe (recipe-prior-refactor-ka2-v1) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 61.975; mechanism `structural`.



Selected ka2 recipe passes interp_img_patch_shift (5/5 terminal checks: interp_rmse=0.0367, interp_peak=0.9149, interp_hq=0.9833) and nearly reaches saddle_barycentric (endpoint_surface_mae=0.0545, barycenter_surface_mae=0.0672, barycenter_hq=0.7639 vs 0.78), while cutting secants on sparse_arc (interp_hq=0.5152).

**Next:** Evaluate whether combining KA2 asymmetric critic updates with local latent transport closes the 0.016 barycenter_hq gap on saddle_barycentric.

[Record and receipt provenance](records/readout-a2a29effc3b7a25d04c3f822.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-a324737f266da9156ed6ed2f

**Scope:** research_diagnostic; scientific; revision `910c746cc18347ce72d108dcef81f283fc5f2efd2117fc6358f7b485ba172ae1`.

A globally smaller normalized learning rate with positive cosine terminal floors stabilizes conditional-energy DDGAN calibration while preserving its stochastic distribution objective and genuine adversarial game.

**Observed:** {'FAIL': 2}; wall seconds 1134.834; mechanism `structural`.



The matchedGPU0 baseline bimodal completed all5000updates and FAILS its original conditional, marginal, local mean/spread and terminal-window gates. The Gaussian baseline was reused exactly without another paid run.

**Next:** Retain this complete matched baseline and both original600s incomplete attempts. Compare the registered final one-field objective-geometry recipe on unchanged full tasks; no baseline rerun, gate relaxation or qualification.

[Record and receipt provenance](records/readout-a324737f266da9156ed6ed2f.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v10 · readout-a3aa33cc909a1545b2341f1c

**Scope:** current; scientific; revision `8d8231faadfd8398b949d80d3de4300ae193089b3b6c582912db602f040c55d5`.

Pairing a dedicated schema-3 inter-mode valley 96th-quantile precision thermal residual channel on continuous priors (rhbi_tail_spill_gain=1.8) with wide-span H_0-connected macro-simplicial manifold settling on discrete particle codebooks (rhbi_simplex_macro_settle=true) under one global clock-free, schedule-free, shape-gate-free recipe converts both counted vector_anisotropic (component_covariance_error <= 0.85, sw1_normalized <= 0.18, mass_tv <= 0.15, hq >= 0.85) and stretch interp_saddle_barycentric (all 5 terminal checks PASS) while preserving ring16_acquisition, mode_hold, gaussian1d_stability, precision_gaussian_fp32/mxfp8/nvfp4, interp_img_patch_shift, discrete image tasks img_blobs4/img_intensity2/img_stripes2, and all prior RHBI view passes on protocol seed 0.

**Observed:** {'FAIL': 2, 'PASS': 14}; wall seconds 986.031; mechanism `structural`.

model_ae_gan_multimodal: modes=4, precision=0.94312, mass_tv=0.029297; ring16_acquisition: modes=16, hq=0.97217, mass_tv=0.064941

inconclusive_tier1_blocked: rhbi-v10 passed 15/16 executed Tier 1 tasks (including ring16_acquisition, precision_gaussian_fp32/mxfp8/nvfp4, and clockfree_audit_measurement_v1), and model_ae_gan_multimodal passed all 8 terminal metric bounds at step 1200 (precision=0.9431 >= 0.90, component_center_rms_sigma=0.9560 <= 1.0) but missed the 5-checkpoint terminal stability suffix by 1 step (passing_suffix=4/5, component_center_rms_sigma=1.0017 at step 1000) because rhbi_tail_spill_gain=1.8 held thermal scale s_t warm too long against uncooled autoencoder reconstruction gradients.

**Next:** Advance to rhbi-v11 by calibrating rhbi_tail_spill_gain from 1.8 to 0.95 while holding rhbi_simplex_macro_settle=true and all other recipe fields fixed, restoring early post-acquisition thermal lock on model_ae_gan_multimodal while retaining valley bridge clearance on vector_anisotropic.

[Record and receipt provenance](records/readout-a3aa33cc909a1545b2341f1c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-a3ec7ffcdaaf0f88b4171cec

**Scope:** current; scientific; revision `bb975f7fe7f68c1e638cf5a36973a125d34fe4285aad773056e4b969fdf6cecc`.

Ordinary re-qualification of the unchanged selected release07-gan-v3 recipe on the current develop source reproduces its recorded discriminator_stability outcomes.

**Observed:** {'BLOCKED': 1, 'FAIL': 3, 'INCOMPLETE': 5, 'PASS': 5}; wall seconds 252.911; mechanism `structural`.

ae_gan_hold: recon_mse=0.0038224

Ordinary discriminator_stability revision-14 re-qualification reproduces all 7 core Tier 1 verdicts (5 PASS, ring16_acquisition FAIL, clockfree_audit_measurement_v1 FAIL) and stops at Tier 1; revision-9 qualification retained.

**Next:** Retain revision-9 qualification in technique-inventory; address ring16_acquisition and clockfree_audit_measurement_v1 in family follow-up.

[Record and receipt provenance](records/readout-a3ec7ffcdaaf0f88b4171cec.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-a66096228357c6881cb90290

**Scope:** research_diagnostic; scientific; revision `4b199ad28279ef9616e67ebc460824f614dc141a7f00ca739993218fc5635ae6`.

One global projected transport auxiliary repairs sustained independent-prior generation in hard VAEGAN while retaining the categorical positive reference; measure an exact matched CUDA baseline before causal comparison.

**Observed:** {'FAIL': 2}; wall seconds 452.695; mechanism `structural`.



Matched CUDA VAEGAN baseline completes both1200-update tasks but fails sustained gates: hard2/5 and categorical3/5 terminal checks; both endpoints pass.

**Next:** Retain this baseline, compare only bounded registered substantive repairs, and confer no ordinary qualification or default credit.

[Record and receipt provenance](records/readout-a66096228357c6881cb90290.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### e22 · readout-a6c971db57845c4e9efbaa1b

**Scope:** research_diagnostic; scientific; revision `abdd47cc9a0817e34adb608e3906dd63f3220742247ae81ed83d45c9e8a7e411`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.564; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-a6c971db57845c4e9efbaa1b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v6 · readout-a8bc8325d7857fdf467674af

**Scope:** current; scientific; revision `26abf8d916172cd34a4ee0ce10c376a3e4a1e9ffc80b03f126ae68d8e0a3b57f`.

Coupling Born-Infeld Lipschitz softplus sharpness settling directly to the 5-channel thermal equilibrium scale via rhbi_settle_mode='thermal' (eff_beta = 12.0 + 16.0 * (1 - s_thermal)^2, strictly inactive while s_thermal=1.0 during pre-equilibrium mode acquisition) alongside Poisson-Weibull 1D AKT horizon normalization (rhbi_akt_mode='weibull') and false-shock-proof thermal reopening (rhbi_thermal_shock_mu=0.35, rhbi_fibre_rel_blend=0.0) eliminates all four rhbi-v5 pre-equilibrium regressions, preserves 27/27 Tier-1 and 82/115 counted ordinary view passes, and unlocks stretch precision_gaussian_nvfp4 (cdf_ks <= 0.05) alongside precision_gaussian_mxfp8 under one global clock-free, shape-gate-free recipe on protocol seed 0.

**Observed:** {'FAIL': 12, 'PASS': 26}; wall seconds 3089.045; mechanism `structural`.

img_stripes2: modes=2, hq=1

rhbi-v6 passes 26/38 executed tasks (0 BLOCKED) on ordinary discriminator_stability CUDA seed-0 qualification on NVIDIA RTX A6000: 13/13 Tier 1 tasks PASS (11/11 required + 2/2 diagnostic = 27/27 Tier 1 view requirements, including precision_gaussian_fp32 cdf_ks=0.02657), 10/18 required Tier 2 tasks PASS (55/78 Tier 2 view requirements, 82/115 total counted view requirements), and all three 1D precision tasks PASS including stretch precision_gaussian_mxfp8 (cdf_ks=0.02215) and precision_gaussian_nvfp4 (cdf_ks=0.03763, std_ratio=1.01832).

**Next:** Promote rhbi-v6 as the canonical RHBI selection, merge latest origin/develop, and refresh current Forge readouts and uncontended DCGAN scaling table.

[Record and receipt provenance](records/readout-a8bc8325d7857fdf467674af.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p--01eca360219ea5225a6e30a31800c7bbe70ca08c3800e259c17a67f0ea528ab5 · readout-a91dbccf086504f9accaee6b

**Scope:** research_diagnostic; scientific; revision `3041a40a01932d40c304875add3e4e2689f4e535636861f3319a1a1f6792c402`.

Within the existing input-noisy/output-clean full-horizon K3P recipe, slower critic rates and faster learned-prior transport may reduce ring within-mode covariance error. Positive critic coefficients .5/1 test this tradeoff while each complete global recipe must pass all five ordinary Tier 1 tasks.

**Observed:** {'PASS': 2}; wall seconds 16.21; mechanism `floor_constant`.



Both800-update diagnostic variants PASS movement/slope: original-schedule1.0542289019 and stretched-schedule .9759703279. These are diagnostic passes only and grant no ordinary qualification.

**Next:** See reports/forge/k3p-two-pole-horizon-v1/README.md and saved force analysis. Retain the main gate and family selection. Investigate existing global critic/particle balance before any new separately bounded search; no automatic follow-on work.

[Record and receipt provenance](records/readout-a91dbccf086504f9accaee6b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v1 · readout-a94316e3e91606df03a346de

**Scope:** current; scientific; revision `3e717461929e3760dcb3d4b87b6452a001c37d644e642179fda7bd2fa3752a8c`.

One global clock-free RHBI configuration (debiased slack-screened unbalanced Born-Infeld Sinkhorn transport + endpoint gradient cap + variational equilibrium cooling) executes cleanly across scalar, conditional, joint BiGAN, AE/VAE and multimodal particle/network hosts on protocol seed 0.

**Observed:** {'FAIL': 7, 'PASS': 8}; wall seconds 1628.459; mechanism `structural`.

ae_gan_hold: recon_mse=0.0040893; model_ae_gan_multimodal: modes=4, precision=0.98486, mass_tv=0.074463

On ordinary discriminator_stability Tier 1 under protocol seed 0 on NVIDIA RTX A6000 (commit d2abe7afb), untuned rhbi-v1 completed all 15 Tier 1 tasks cleanly (0 BLOCKED, 0 INCOMPLETE, 0 ERROR): 8 PASS (7/11 required: ae_gan_hold, five_word_joint_smoke, gaussian1d_smoke, model_ae_reconstruction, two_pole, unused_token_hold, zero_tanh_smoke; plus 1/2 diagnostic: clockfree_audit_measurement_v1) and 7 FAIL (4/11 required: ring16_acquisition, model_ae_gan_multimodal, model_vae_categorical, model_vae_hard; 1/2 diagnostic: precision_gaussian_fp32; 2/2 stretch: precision_gaussian_mxfp8, precision_gaussian_nvfp4), satisfying its preregistered ae_gan_hold prediction (hold=0.00535 <= 0.025, recon_mse=0.00409) and stopping later tiers at the Tier 1 prerequisite barrier.

**Next:** Pin rhbi-v1 as the selected canonical RHBI family candidate in configs/forge/selections/family-current-v1.json, register its frozen numerical snapshot and compact receipts, regenerate reports/forge/technique-inventory.md, and test Kantorovich-slack residual damping below DualNorm smoothing plus SBNP low-mobility prior scaling (prior_lr_mult=0.02) in a follow-up rhbi-v2 study.

[Record and receipt provenance](records/readout-a94316e3e91606df03a346de.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-r1r2-matched-v1 · readout-a9c44ece3a8e2265db452db4

**Scope:** current; scientific; revision `0a2b2de1a8a75005f8f76ca17130010f477c520425761cff04dd2a4f81f2bc3a`.

Replace only the critic penalty by frozen historical r1r2 while retaining the K3P optimizer, noise, A2, host and sampling contracts.

**Observed:** {'FAIL': 1}; wall seconds 5.812; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 0.06404702365398407, 'mean_abs': 0.1786370873451233}. One completed attempt cost 5.811797590 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/87f406b78184409396a26f61680a5da0.json) · [Record](records/readout-a9c44ece3a8e2265db452db4.json)

### recipe-prior-refactor-release07-gan-v3-v1 · readout-abb6ed1e88edf492ed9514f5

**Scope:** research_diagnostic; scientific; revision `46f1fa3af56a91b94c4c67961f9860e26491ef6c6186622d4cd8351cf81ac485`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.294; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-abb6ed1e88edf492ed9514f5.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-no-output-noise-diagnostic · readout-abd06840fa1f42a4f64c13ab

**Scope:** calibration_diagnostic; scientific; revision `0fa6918ca4d0456515de09dbcfaeedffba1d4cecaab00dee5028bcf3b0b34603`.

Removing additive training output noise may prevent the clean learned-MoG native distribution from contracting late. This is a one-factor diagnostic of the training/scoring-law mismatch, not a predicted positive or a full-reference claim.

**Observed:** {'FAIL': 1}; wall seconds 93.747; mechanism `structural`.

grid100_affine_square_named_v1: precision=0.98165, center_rms_sigma=0.18834, mass_tv=0.0317, radial_ks=0.06713

One preregistered full 7000-update clean learned-MoG grid diagnostic FAIL for 93.746687201 paid seconds. Final 100 modes do not compensate for sustained coverage FAIL (one final coverage check) and 0/5 joint terminal passes. The independent 100k live holdout fails abs covariance bias .1630054 > .10 and radial KS .06713046 > .04; centre RMS .1883405 and mass TV .0317 pass. EMA holdout also fails. All intended G/D/prior updates completed, state finite and no unintended RNG deviations. No full-reference positive, accepted calibration or ordinary qualification.

**Next:** Stop this exact diagnostic and its infeasible no-output-noise-reference-v1 roster: every lineage is already smoke-negative or reference-negative, so no possible completion satisfies a reference positive and zero false rejects. Do not spend on its smoke, other reference cells or 14k continuation. Preserve all 16 reference purposes and frozen criteria. Next stage should first use saved terminal artifacts to locate per-mode shape/centre/critic-response failure and require a new supported structural hypothesis with a separately bounded preregistration. Production promotion remains blocked on accepted calibration and finished exact qualification.

[Record and receipt provenance](records/readout-abd06840fa1f42a4f64c13ab.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-acc70fcaf6cedb9a3772e41e

**Scope:** calibration_diagnostic; scientific; revision `229770eceb2d51236085562985c341033df825c4c85fc20adbebdef313f8c65b`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 1, 'PASS': 5}; wall seconds 37.763; mechanism `structural`.



Frozen v2 smoke remains FAIL (two_pole travel); unused_token_hold and learned-MoG ae_gan_hold PASS. Three independent CPU reference tasks, trajectory/residual_student/unipolar, now PASS their complete live terminal windows for 20.864916388 CPU wall seconds. Reference remains UNKNOWN: only 3/16 cells measured; these are declared cloud/nonsampled controls, not MoG quality evidence.

**Next:** Stop filling the v2 matrix solely for adoption. Preserve the remaining 13 baseline reference cells and both ablation reference vectors as unknown. Preregister a separately versioned test of the already-proposed quick-discriminator screen with unchanged thresholds and the same 16 independent references, bounded GPU stages and no automatic promotion. Keep the registered physical GPU pilot for its distinct operational question when ownership is available.

[Record and receipt provenance](records/readout-acc70fcaf6cedb9a3772e41e.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-ad21627bf4def92894dd98ce

**Scope:** research_diagnostic; scientific; revision `3e8ee59575d1b9599fc0eced9247ef1ed2a369956516eba9baf0b21360c9fd6e`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 164.039; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/bcap-dualnorm-dcgan_performance_b64-addedac8ac836908.json) · [Record](records/readout-ad21627bf4def92894dd98ce.json)

### empf-v5 · readout-aef7cbc33a8a6b3326b86bb8

**Scope:** current; scientific; revision `a457aa4e6487d00f5630eeebf70830674145a1dae9f64dc38d6643f6eff3508b`.

Combining empf-v4's fixed K = min(max_prototypes, B) small-batch prototype banks, stationary zero-shock normalization, and fused autograd syncs with empf-v3's thermal_decay = 0.92 transport smoothing, cool-gated Born-Infeld batch_gain, and exact Sinkhorn-free discrete-simplex Chamfer residuals clears all Tier 1 gates (including model_vae_categorical, model_vae_hard, ring16_acquisition, model_ae_gan_multimodal, and precision_gaussian_fp32/mxfp8) and advances through Tier 2/3 to realize gaussian1d_stability, target_shift_recovery, ring_hold, and ring_extension gains.

**Observed:** {'FAIL': 4, 'INCOMPLETE': 5, 'PASS': 24}; wall seconds 5651.411; mechanism `structural`.

ae_gan_hold: recon_mse=0.0043931

empf-v5 passed all 12 non-word counted Tier 1 gates (including model_vae_categorical, model_vae_hard, ring16_acquisition, model_ae_gan_multimodal, and precision_gaussian_fp32) and passed all completed checks on five_word_joint_smoke through step 17,501/20,001, but hit the 900s wall-time cap (INCOMPLETE) under concurrent multi-agent GPU load due to 9 per-step serial cuSOLVER SVD calls on dense 2D matrices.

**Next:** Introduce declared optimizer_dense_profile='newton_schulz_v1' in empf-v6 using DualNorm's guarded 32-iteration cubic Newton-Schulz polar solver (_newton_schulz_polar + _PolarGraphCache) for 2D dense matrices while keeping the empf-v5 formulation unchanged.

[Record and receipt provenance](records/readout-aef7cbc33a8a6b3326b86bb8.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### release07-gan-v3-task-adapted-v1 · readout-afd6d27c7d1eae64aa13299f

**Scope:** current; scientific; revision `d5341d8d12528f9e34030480544dd84fa7b933eed8474ff19e568dfb56b26cbc`.

Make the released v0.7 optimizer and BCap training recipe executable across frozen toy hosts through explicit task-owned resource and objective bindings; measure ordinary gates without importing native or historical credit.

**Observed:** {'FAIL': 1}; wall seconds 5.508; mechanism `structural`.



The released v0.7 task adaptation passes preflight for all24 ordinary requirements and executes through the frozen public component host. Complete80-update two_pole FAIL: mean_abs0.29372087121009827<0.30; grad_med0.3106328547000885<=1 PASS. All24 observations and the required suffix fail the joint gate. One CPU attempt cost5.508408382942434 paid seconds; no execution errors. Ordinary scheduling stopped; remaining23 cells UNKNOWN, no integration blockers on the successor.

**Next:** Stop this exact failed revision; keep the integration and reusable host adaptation. Review the provisional tier1 movement/convergence protocol before another training hypothesis. Any post-failure quality comparison requires a separately registered bounded diagnostic lane; no threshold change, seed repeat, continuation, automatic matrix filling or default adoption follows.

[Evidence](../../reports/forge/technique-receipts/df1794ebea6444c5a10c2a953843257d.json) · [Record](records/readout-afd6d27c7d1eae64aa13299f.json)

### recipe-prior-refactor-ka2-v1 · readout-b03e5461a1e254ae43817e7a

**Scope:** research_diagnostic; scientific; revision `84ffb37f3eb56de7405cfd6260649c805c1d7f8d231ad207a6e95680780f6edc`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.608; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-b03e5461a1e254ae43817e7a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-b14773cc578e463ee6199a5a

**Scope:** research_diagnostic; scientific; revision `d8d32287e6863a2f3f4a28ef7c88853d47870ed0709a23fdf253522e11d96a62`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 66.051; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-b14773cc578e463ee6199a5a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-vaegan-transport-v1 · readout-b194dccb65688ca056129011

**Scope:** research_diagnostic; scientific; revision `0612a7116b3d77a9724d9e6286115a6bebf369d011f4ca85e592d5a8d337b306`.

One global projected transport auxiliary repairs sustained independent-prior generation in hard VAEGAN while retaining the categorical positive reference; measure an exact matched CUDA baseline before causal comparison.

**Observed:** {'FAIL': 2}; wall seconds 468.822; mechanism `structural`.



Transport alone fails both unchanged VAEGAN tasks with0/5 terminal passing checks; final likelihoodHQ hard.8191/categorical.6858. Reconstruction stays informative but independent generation degrades.

**Next:** Stop this exact revision; keep failure in the current comparison. Its separately frozen finite-budget settling successor is measured without changing gates, source, initialization or data.

[Record and receipt provenance](records/readout-b194dccb65688ca056129011.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-global-repair-v1 · readout-b214c23e088f39337c3d8784

**Scope:** current; scientific; revision `9bc3da4de808300aaa74effda23f1114910ce2c819bda3fcd579bb8b6d5f5e8e`.

The exact repaired global recipe acquires the five-word inverse task and transfers without task-specific optimizer tuning across every eligible required discriminator-stability task under ordinary prerequisites.

**Observed:** {'FAIL': 1}; wall seconds 5.653; mechanism `structural`.



Ordinary global candidate failed two_pole movement: mean_abs=0.0468712635<0.3; qualified tier0. Other25taskcells, including word, are UNKNOWN.

**Next:** Initial complete candidate rejected. Its subsequent separately bounded global coupled-rate search is concluded; no automatic paid continuation or default adoption.

[Record and receipt provenance](records/readout-b214c23e088f39337c3d8784.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-b389e250ac5b3ad0763d2c82

**Scope:** current; scientific; revision `3e568ed2213479cc8bbd8844796ab6f0ca4c804fa846e627563b06f556001089`.

Ordinary discriminator_stability measurement of atlas-particle-normal-v4 (particles + NormalNoiseManipulator(sigma=0.0065), birth_death_cells=256, clean fixed output noise 0.0, d_guard_ratio=0.0, KA2 pre-anchor critic-only LossSurprise guard) under protocol seed 0 on NVIDIA RTX A6000 completes every required Tier 1 cell cleanly and reproduces its 11/13 required Tier 1 passes (plus clockfree_audit_measurement_v1 diagnostic PASS) for selection as the Atlas family candidate.

**Observed:** {'PASS': 21}; wall seconds 446.773; mechanism `structural`.

ae_gan_hold: recon_mse=0.0099752

On ordinary discriminator_stability Tier 1 under protocol seed 0 on NVIDIA RTX A6000, atlas-particle-normal-v4 completed all 14 Tier 1 tasks cleanly (0 INCOMPLETE) and reproduced 14/14 Tier 1 verdicts identically to prior-ablation-full-suite-v1 (readout-423f82f13ee30c0a95fd266a): 8 PASS (7/13 required: gaussian1d_smoke, unused_token_hold, ae_gan_hold, zero_tanh_smoke, model_ae_reconstruction, model_vae_categorical, model_vae_hard; plus 1/1 diagnostic: clockfree_audit_measurement_v1) and 6 FAIL (two_pole, ring16_acquisition, five_word_joint_smoke, model_ae_gan_multimodal, model_ddgan_gaussian_chain, model_ae_ddgan_gaussian_chain), stopping later tiers at the Tier 1 prerequisite barrier.

**Next:** Promote atlas-particle-normal-v4 as the selected canonical Atlas candidate in configs/forge/selections/family-current-v1.json, register its frozen numerical snapshot and compact receipts, regenerate reports/forge/technique-inventory.md and family pages, and investigate two_pole/ring16_acquisition and joint-inference (BiGAN/DDGAN/five_word_joint) Tier 1 failures in follow-up studies.

[Evidence](../../) · [Record](records/readout-b389e250ac5b3ad0763d2c82.json)

### r1r2--724b3d52fbb2172a985d1bf29890b2ef9ba2598c9f1724d93a9afa7c872cbdad · readout-b42a8f277178ec58808d6b5c

**Scope:** current; scientific; revision `98d01c8b382226c46c312202c5a952ad5d37355192778c149090a03b8fd17bfc`.

Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.

**Observed:** {'FAIL': 1, 'PASS': 3}; wall seconds 30.974; mechanism `floor_constant`.

ae_gan_hold: recon_mse=0.0035668

This exact global configuration passed two_pole, unused_token_hold and ae_gan_hold, then failed ring16 quality. Tier1 is3/5; word and22remainingcells are UNKNOWN. No configured standard or default adoption.

**Next:** Bounded global search concluded. Retain the exact historical family incumbent as unqualified evidence. No further paid stage is authorized; require a separate bounded global proposal before any continuation.

[Record and receipt provenance](records/readout-b42a8f277178ec58808d6b5c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-b7acc4d7ec8bb9cb7dd704a8

**Scope:** research_diagnostic; scientific; revision `6bc63bbe693a22f5aada19b0855b56d81fd389db4ec9275783679368e0058df3`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.905; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-b7acc4d7ec8bb9cb7dd704a8.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-ba142a8baa78fa77a792bf27

**Scope:** research_diagnostic; scientific; revision `51786f37adc7cbd5d6ac2d1943e7838af3076ef066cb727f87a3425ae1079e6b`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.852; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-ba142a8baa78fa77a792bf27.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-prior-frozen-v1 · readout-bb93e156247cbef1aa781703

**Scope:** research_diagnostic; scientific; revision `21e3bfccfbf3b99d397b160ef62df3654082414ba1068129101254d6e17ef6e8`.

Freezing the particle table (same initial locations and capacity) loses sampled-latent passes that the learned BISPF default holds.

**Observed:** {'BLOCKED': 1, 'FAIL': 18, 'INCOMPLETE': 4, 'PASS': 13}; wall seconds 2557.195; mechanism `structural`.

img_bars4: modes=2, hq=0.78125

Frozen table: 13 PASS / 18 FAIL of 49; loses ring16_acquisition (hq 0.61), mode_hold, ae_gan_hold and ring_extension vs learned; crashes trajectory/residual_student hosts. Lower mass TV but precision 0.03-0.05 on the native grids. See reports/prior-ablation-2026-10-25/README.md.

**Next:** Stop; use as the matched control for a learned-offset-on-frozen-anchor structural idea.

[Record and receipt provenance](records/readout-bb93e156247cbef1aa781703.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-bbadc8452e77fa75caff13c6

**Scope:** research_diagnostic; scientific; revision `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`.

The global quintic rule retains the native BISPF pass set on both unchanged width-64 image tasks, including a positive stripes acquisition pass, under the same seed, public initializer, data sequence, prior, sampling, 600 updates and 24 checks.

**Observed:** {'PASS': 2}; wall seconds 40.113; mechanism `structural`.

img_stripes2_w64_optimizer_diagnostic: modes=2, hq=1; img_bars4_w64_optimizer_diagnostic: modes=4, hq=1

Quintic convolution rule passes both width64 gates at600 updates: stripes HQ1 with2 modes and12 terminal passing checks; bars HQ1 with4 modes and9 terminal passing checks. Both exceed the required5-check stable suffix. All optimizer groups complete600 updates and all states remain finite.

**Next:** Proceed only with separately frozen matched full DCGAN performance and ordinary quality campaigns from H2. Selection requires the declared latency targets and preservation of every incumbent ordinary pass; no public-default adoption follows this diagnostic.

[Record and receipt provenance](records/readout-bbadc8452e77fa75caff13c6.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-bbd7f8cdac1d2a717c806681

**Scope:** research_diagnostic; scientific; revision `7b2fcfc5648bad6eb83bd86214bff1b38acd2c01e870db3a79998406444c4681`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.158; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-bbd7f8cdac1d2a717c806681.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r3gan-stacked-training-toy-v1 · readout-bc440d0469187c9299bd524c

**Scope:** current; scientific; revision `724a672468491e40b3f214c3b44b8913b9317266346527139da7a1de750d8d44`.

Transfer the Modern GAN paper's Stacked MNIST training recipe to Forge's fixed toy hosts, distinguishing the complete scheduled Adam/R1/R2 recipe from the prior penalty-only swap.

**Observed:** {'FAIL': 1}; wall seconds 5.347; mechanism `structural`.



The corrected Modern GAN Stacked MNIST training recipe, adapted to the fixed Forge toy hosts, failed the complete 80-update two_pole movement gate: mean_abs=0.011449367739260197 <0.30; grad_med=0.13321508467197418 <=1 passed. 24 observations, zero passing observations/suffix; actual Adam beta2 0.9 to0.99, gamma1 to0.1, constant LR0.0002,80 penalty applications, no interventions. One CPU attempt cost5.347339480998926 paid seconds. All23 remaining requirements are unknown; no execution error or downstream spending.

**Next:** Stop this exact failed revision. Preserve full denominators and the separate recorded cohorts; no seed repeats,unchanged reruns or downstream gate filling. Before a substantive further test, establish an independently justified and bounded convergence/reference protocol; calibrate the provisional screen before adoption.

[Evidence](../../reports/forge/technique-receipts/ea66e44192c048eaa97f8595c5e55508.json) · [Record](records/readout-bc440d0469187c9299bd524c.json)

### recipe-prior-refactor-bcap-pure-v1 · readout-c0eb323419b0be6463f34d1b

**Scope:** research_diagnostic; scientific; revision `f1f5aa34d333c7c3874e81b718103c25a0fc255b10eb4ed28268e192bba3af22`.

Evaluate whether unchanged selected bcap-pure recipe (recipe-prior-refactor-bcap-pure-v1) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 3}; wall seconds 84.732; mechanism `structural`.



Selected bcap-pure recipe fails all 3 zero-sigma ParticlePrior interpolation tasks (0/3 PASS: sparse_arc interp_hq=0.4127, img_patch_shift interp_rmse=0.1490/interp_peak=0.3015, saddle_barycentric barycenter_hq=0.3000).

**Next:** Investigate BCAP gradient-cap interaction with unsmoothed zero-sigma ParticlePrior updates before proposing a recipe-owned latent regularizer.

[Record and receipt provenance](records/readout-c0eb323419b0be6463f34d1b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### rhbi-v7 · readout-c162cb8fb6ab4d55620a8693

**Scope:** current; scientific; revision `af74e8b4a3d6f9a68c4fc8a6f87c808172f6769788da1a74828e67de99c468bd`.

Masking off-diagonal discrete-particle batch clones in symmetric Feydy self-coupling (rhbi_self_coupling='feydy_unique') alongside Pareto-protected Weyl-stratified detached-vertex k-NN 1-simplex/2-simplex Born-Infeld manifold regularization (rhbi_simplex_weight=0.35) and hollow multi-scale IMQ conditional fiber MMD (rhbi_cond_ot_mode='imq_cmmd', rhbi_cond_ot_weight=1.5) unlocks held discrete image benchmarks img_blobs4 and img_intensity2 (hq >= 0.90) while preserving img_stripes2, all 27/27 Tier-1 and 55/78 Tier-2 counted view requirements, and all 3/3 1D precision benchmarks (FP32, MXFP8, NVFP4) under one global clock-free, shape-gate-free recipe on protocol seed 0.

**Observed:** {'FAIL': 15, 'PASS': 26}; wall seconds 3074.341; mechanism `structural`.

ae_gan_hold: recon_mse=0.0053954

rhbi-v7 passes 26/41 executed tasks (0 BLOCKED) on ordinary discriminator_stability CUDA seed-0 qualification on NVIDIA RTX A6000, confirming the study prediction on img_blobs4 (hq=0.96875 >= 0.90, modes=4) and newly passing img_intensity2 (hq=0.90625 >= 0.90, modes=2) while preserving img_stripes2 (hq=1.0) and all 82/115 counted view requirements (27/27 Tier 1, 55/78 Tier 2), but output-space zero-distance masking in feydy_unique conflates discrete latent sampling clones with 4-bit/8-bit quantized forward plateaus, regressing stretch precision_gaussian_mxfp8 (std_ratio=1.1052 > 1.10) and precision_gaussian_nvfp4 (cdf_ks=0.0777 > 0.05).

**Next:** Advance to rhbi-v8 by binding feydy_unique clone deduplication strictly to exact latent-input clones (z_i == z_k) rather than quantized output collisions (x_i == x_k), preserving both img_blobs4/img_intensity2 and MXFP8/NVFP4.

[Record and receipt provenance](records/readout-c162cb8fb6ab4d55620a8693.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-c164627eb57d4e7515314cd6

**Scope:** research_diagnostic; scientific; revision `ccf1ac64bec1d00ef53c680605370201505da3b53d007bfda3d4a4c8c21ce178`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 165.875; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-c164627eb57d4e7515314cd6.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-misgan-amortized-v1 · readout-c21940571fd7349e5f2a9c1b

**Scope:** research_diagnostic; scientific; revision `43489bc17bc07057ac6f5b53d3368f30827498bd4b1415badfcbff0202fc519e`.

Retain BISPF core and distinguish incomplete-data recovery from learned multimodal conditional amortization with the frozen scorer and controls.

**Observed:** {'FAIL': 1}; wall seconds 615.736; mechanism `structural`.

model_misgan_four_mode: modes=4, hq=0.99463, mass_tv=0.16968

Full scoped four-mode run completed1200updates/24checks and FAILed all5terminal complete gates. Endpoint generatorHQ=0.994629, massTV=0.169678, ambiguousHQ=0.982666, branchTV=0.143311. Preserve every terminal failed bound; no ordinary or historical100-mode pass credit.

**Next:** Stop this exact revision. The sole bounded composite masked-quantile transport plus cosine-settling followup changes the failed mass/settling mechanisms while keeping the frozen task, initializer, prior, sampling and1200update protocol.

[Record and receipt provenance](records/readout-c21940571fd7349e5f2a9c1b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-v2 · readout-c2d279f7d47ac009ee80d771

**Scope:** current; scientific; revision `96f0f8a2aee0663f0fe92e49641deb107b81201ae0efea33d9a54e24aaba8967`.

BISPF v2 repairs at least 6/11 Tier 2 failures on discriminator_stability reaching >= 16/21 passes while preserving all 7 Tier 1 passes (22/22 across views) and all 10 passing Tier 2 tasks with 0 regressions.

**Observed:** {'FAIL': 11, 'PASS': 19}; wall seconds 2565.992; mechanism `structural`.

ring16_acquisition: modes=16, hq=0.9519, mass_tv=0.092041

Born-Infeld Sinkhorn Particle Flow v2 (BISPF v2, bispf-v2) achieves full qualification through Tier 3: passing 7/7 Tier 1 tasks (100% across all 6 views), 10/10 required Tier 2 baseline tasks with 0 regressions, and Tier 3 endurance qualification with ring_hold (1200/1200 checks, 0 failures) and ring_extension (300/300 checks, 0 failures) PASS. Recomputed complete live curves and terminal suffixes confirm stability under direct thermal hook scaling, SBNP covariance tethering, and late-training harmonic mode capture.

**Next:** Promote bispf-v2 to repository-wide default selection in family-current-v1.json, refresh publication records, and compile final memory.

[Evidence](../../reports/forge/bispf/README.md) · [Record](records/readout-c2d279f7d47ac009ee80d771.json)

### rhbi-v9 · readout-c2f6f3c483dff61d72296842

**Scope:** research_diagnostic; scientific; revision `9e81b5afc9298a9cea9a90952637fc73b11ca93f518c0777145ec19ce1c6efd8`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 69.771; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for rhbi-v9 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-c2f6f3c483dff61d72296842.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-c3df7159c90725fdef1cb700

**Scope:** calibration_diagnostic; scientific; revision `ef34dc1682c082151c202c4549af6343fa8b309ac5cf71c8ad29098dc55e77e4`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 36.238; mechanism `structural`.

img_intensity2: modes=0, hq=0; vector_two_broad: hq=0.98828, mass_tv=0.00024414

Corrected intensity FAIL and learned-MoG vector_two_broad PASS under the same frozen v3 source. The physical two-GPU workflow pilot passed; scientific calibration remains blocked.

**Next:** Stop automatic reference-matrix expansion. Audit canonical host architecture provenance and register any justified substantive diagnosis under a new frozen identity. Preserve full 16 independent-reference denominator; unknown tasks are neither passes nor failures. Keep legacy cutover blocked until accepted calibration.

[Record and receipt provenance](records/readout-c3df7159c90725fdef1cb700.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-c4561c13c0ea2d2622f5c963

**Scope:** research_diagnostic; scientific; revision `255bb46b0d2a4037d9a53301d6800335c01d73a7689d30493b4918733709af9e`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 67.168; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/r1r2-dcgan_performance_b64-d64d96234ed8a651.json) · [Record](records/readout-c4561c13c0ea2d2622f5c963.json)

### e22 · readout-c66861b42a3568d59f7c8b56

**Scope:** research_diagnostic; scientific; revision `73c3876a42f7c613680e3cf02d00934fb6cec8495fb83d45b5c6a0099a3b731f`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 65.094; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for e22 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-c66861b42a3568d59f7c8b56.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-c723a88e5122f97011aabad9

**Scope:** calibration_diagnostic; scientific; revision `5d60f405c372f33d8c96675c772fe97e10bb922cfe313d8e66722de9d32bc289`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 1, 'PASS': 2}; wall seconds 16.495; mechanism `structural`.

ae_gan_hold: recon_mse=0.0045745

All three registered smoke cells completed at the frozen current cohort: two_pole FAIL (mean_abs0.1065209<0.30; grad_med0.0637243<=1), unused_token_hold PASS, learned-MoG ae_gan_hold PASS. The entire smoke decision is FAIL. Independent quality reference cells remain unmeasured.

**Next:** Retain all nine cheap-cell outcomes and costs. Keep ordinary qualification stopped. Review the exact historical/public-host parity differences before deeper reference spending; reserve the required GPU window and register only justified missing reference cells. Do not tune thresholds or treat unmeasured reference labels as failures.

[Record and receipt provenance](records/readout-c723a88e5122f97011aabad9.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v7 · readout-c74fc1dbb493da20f3e37f95

**Scope:** research_diagnostic; scientific; revision `53f1c7f075a97c6265c1951eb687a12f43296033d33e7317445c133be235c949`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.521; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for empf-v7 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-c74fc1dbb493da20f3e37f95.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-ddgan-energy-v1 · readout-c7a4fe6fbfe14b4e43558471

**Scope:** research_diagnostic; scientific; revision `0179185c13fe852b4c1d86796abef1850412345fbe26aa96b4fcf234e470ff5b`.

A conditional energy score added to BISPF improves low-noise conditional-law calibration; the matched GPU0 selected-recipe control may retain the historical Gaussian failure.

**Observed:** {'FAIL': 1, 'INCOMPLETE': 1}; wall seconds 1090.469; mechanism `structural`.



Gaussian completed2500updates and FAILS all-gate terminal stability despite endpointCDF improvement; bimodal timed out before5000updates and remains INCOMPLETE.

**Next:** Stop exact revision; execute registered lower-mobility positive-floor energy repair on unchanged scientific tasks, with timeout-only1200s bimodal variant and exact complete Gaussian baseline reuse. No ordinary qualification or default adoption.

[Record and receipt provenance](records/readout-c7a4fe6fbfe14b4e43558471.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2-repair-r1only-20261022 · readout-cd109ec1175f5a26582f9968

**Scope:** current; scientific; revision `704c803e0c8bf425e1c0cd673b77e0332278cd8163ca9058a95c50a2f00fbdab`.

Removing only R2 fake-input gradient pressure repairs the unchanged Ring16 local-density gate while retaining every other required Tier1 gate with one global native R1 recipe. This new-source operational repair fixes snapshot provider dependencies, preserves original runtime errors and routes the already-declared tanh evaluator; all original scientific gates and recipe values remain unchanged.

**Observed:** {'FAIL': 4, 'PASS': 10}; wall seconds 487.193; mechanism `structural`.

ae_gan_hold: recon_mse=0.0084931; five_word_joint_smoke: modes=5, mass_tv=0.018945

Hypothesis falsified: 9/13 required PASS, 4 FAIL; separate clock PASS. Ring covariance worsens to 3.00038088113 and mass TV .1591796875 fails, despite minimum spread improving to .25259119. DDGAN worsens and multimodal AE-GAN generation regresses. Reject R2 removal as a global repair.

**Next:** Retain R2 and the selected historical recipe; no ordinary Tier2 or default promotion. Finish separately scoped own-checkpoint word hold, publish all failures, and require a demonstrated density-force/conditional-tracking prerequisite before another global candidate.

[Evidence](../../reports/forge/r1r2-repair-20261022/README.md) · [Record](records/readout-cd109ec1175f5a26582f9968.json)

### bispf-throughput-v1 · readout-cd9493a6a30096475d381175

**Scope:** research_diagnostic; scientific; revision `5519ffc77cf0cfd46928ca25fbbf29efb9a642d8da333482419f4c9af3a4300d`.

Preserve the two archived incumbent endurance PASS verdicts under their original task protocols, with a separately scoped nonqualifying diagnostic after current revision12 restores four previously held Tier2 failures.

**Observed:** {'PASS': 3}; wall seconds 152.743; mechanism `structural`.

mode_hold: modes=8, hq=0.96924; ring_extension: modes=8; ring_hold: modes=8

The prerequisite mode_hold passes; all 1200 hold checks and 300 extension checks pass at the unchanged first-convergence stop of 3282 updates.

**Next:** Retain the explicit diagnostic scope and original checkpoint/initialization/stream identities; keep current-policy Tier 3 blocked.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-cd9493a6a30096475d381175.json)

### recipe-prior-refactor-ka2-v1 · readout-ce16aac1eef278edef69d071

**Scope:** research_diagnostic; scientific; revision `cd095aefac1ae2d9d8cde29bd32fd18e375669a082e71984cafeddc50e6299b3`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.997; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-ce16aac1eef278edef69d071.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-cecfd17c12d47c1b8f85c63c

**Scope:** research_diagnostic; scientific; revision `19354812348942e7cb8787ec55218bbd7fc3d97d6216b083417b2ab2de60d4e6`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 57.364; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-cecfd17c12d47c1b8f85c63c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-cfb9bf047c6814e0e030457c

**Scope:** research_diagnostic; scientific; revision `3077116cee06431900fca4a10e85287550c6592b5f15497a34659fd1d8a04fe7`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.114; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-cfb9bf047c6814e0e030457c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v4 · readout-d156a482c49f278303ac9880

**Scope:** research_diagnostic; scientific; revision `95cfa5a9ff292735bee035e740d5282399998657fb2f4f04e229cbfef21e4ebb`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 64.879; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-d156a482c49f278303ac9880.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-d230c03f6f71197e5062b283

**Scope:** research_diagnostic; scientific; revision `5061677552fea453f5c9bdaa7a56ba4d0549798824749017eaf6c82e5f88686f`.

Evaluate whether unchanged selected k3p recipe (recipe-prior-refactor-k3p-v1) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 62.752; mechanism `structural`.



Selected k3p recipe passes interp_img_patch_shift (5/5 terminal checks: interp_rmse=0.0381, interp_peak=0.9349, midpoint_novelty=0.4269), and reaches barycenter_surface_mae=0.0768 <= 0.09 (barycenter_hq=0.6974) on saddle_barycentric, while cutting chords on sparse_arc (interp_hq=0.5606, 25/32 bins).

**Next:** Add a recipe-owned latent chord/Jacobian regularizer to bend MLP secant chords onto curved low-dimensional manifolds without MoG.

[Record and receipt provenance](records/readout-d230c03f6f71197e5062b283.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-bcap-pure-v1 · readout-d278a74af331d2f3b7807234

**Scope:** current; scientific; revision `c0f8979dc36bb4f102c4a103087bfc120d71f48ea22f2a5ca52c27d7d06d8557`.

Ordinary re-qualification of the unchanged selected bcap-pure recipe on the current develop source reproduces its recorded discriminator_stability outcomes.

**Observed:** {'BLOCKED': 1, 'FAIL': 4, 'INCOMPLETE': 2, 'PASS': 7}; wall seconds 1014.505; mechanism `structural`.



Ordinary discriminator_stability revision-14 re-qualification reproduces all 7 core Tier 1 passes (including five_word_joint_smoke in 769.0s) and stops at Tier 1 on the revision-14 API model toys and unsupported zero_tanh_smoke; revision-9 qualification retained.

**Next:** Retain revision-9 full qualification in technique-inventory; hold or repair revision-14 API model toys and zero_tanh_smoke separately.

[Record and receipt provenance](records/readout-d278a74af331d2f3b7807234.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### empf-v1 · readout-d46a41a1da2da54d67d154f4

**Scope:** current; scientific; revision `45a5684f2961affd93bfb3cdd35b79928763c60cfe6a3e913492dc1bc77007e4`.

Exponential Moving Prototype Flow (critic_formulation=empf, prior_regularizer=empf) passes Tier 1 (including clockfree_audit_measurement_v1 and all model-family gates), preserves or improves Tier 2 multi-modal and uneven-mass transport under one global constant-LR DualNorm recipe, and passes Tier 3 ring_hold and ring_extension.

**Observed:** {'FAIL': 18, 'INCOMPLETE': 2, 'PASS': 20}; wall seconds 4888.081; mechanism `structural`.

img_blobs4: modes=3, hq=0.90625

On ordinary discriminator_stability (Rev 16) under protocol seed 0 on NVIDIA RTX A6000, untuned empf-v1 passed 11/11 required Tier 1 tasks (100% Tier 1 qualification: gaussian1d_smoke, two_pole, unused_token_hold, ae_gan_hold, ring16_acquisition, five_word_joint_smoke, zero_tanh_smoke, model_ae_gan_multimodal, model_ae_reconstruction, model_vae_categorical, model_vae_hard) plus clockfree_audit_measurement_v1 diagnostic PASS (12/15 Tier 1 total; precision_gaussian_fp32 diagnostic and mxfp8/nvfp4 stretch failed). Advancing to Tier 2, empf-v1 passed 8/18 required Tier 2 tasks (cover_leftover, img_stripes2, mid_scale_identity, residual_student, trajectory, unipolar, vector_spiral, vector_two_broad; 20/38 executed tasks PASS overall), while mode_hold finished at hq=0.8411 < 0.90 (5/8 modes), falsifying the preregistered mode_hold prediction and stopping Tier 3 progression at the Tier 2 barrier.

**Next:** Register empf-v1 as the initial measured selected recipe for the EMPF trainer family on the current leaderboard with no default-adoption claim; in future EMPF v2 studies, scale prototype bank capacity K >= max(64, M) or adaptive split-merge, reduce prototype EMA inertia across distribution shifts, and add timestep-aware conditional prototype routing for multi-step DDGAN chains.

[Record and receipt provenance](records/readout-d46a41a1da2da54d67d154f4.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-misgan-transport-settle-v1 · readout-d57833ecdafd9cda119fefb6

**Scope:** research_diagnostic; scientific; revision `4361bc72b78e2e5cd1fe28820e168a71edf24dcce81f08516986a8666c04c04e`.

The declared composite observed-mass transport and cosine-settling repair passes all twelve unchanged MisGAN recovery and neural posterior bounds for all five terminal checks.

**Observed:** {'PASS': 1}; wall seconds 447.389; mechanism `structural`.

model_misgan_four_mode: modes=4, hq=0.98901, mass_tv=0.093994

PASS on the scoped four-mode MisGAN task: 1,200 updates, 24 post-update evaluations and all five terminal checks pass every unchanged bound. Endpoint mass TV 0.093994, generator HQ 0.989014, ambiguous HQ 0.937256, branch TV 0.078125, branch coverage 1.000000, imputation accuracy 0.972656, observed error 0.0. All mechanism/optimizer/RNG guards pass. No ordinary, original 100-mode, or within-mode covariance qualification.

**Next:** Stop the bounded campaign and retain all failures. Recommend this one complete scoped recipe for review; investigate the same mechanisms on a separately declared larger incomplete-data target before attempting unchanged 100-mode qualification. No seed experiment or automatic default promotion.

[Record and receipt provenance](records/readout-d57833ecdafd9cda119fefb6.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### release07-gan-v3-mog-v1 · readout-d5a99d9470252f695a717dfc

**Scope:** calibration_diagnostic; scientific; revision `a9b2aa0ace96762b024a30c0979a0eedce8d07f9fae5cb5dc03868522132962c`.

Evaluate the full v0.7 GAN v3 dynamics adapted only to the matched affine z2/batch2048/MoG host.

**Observed:** {'FAIL': 1}; wall seconds 75.19; mechanism `structural`.

grid100_affine_paired_laws_v1: precision=0.97504, center_rms_sigma=0.2299, mass_tv=0.08079, radial_ks=0.039679

Preregistered full7000-update grid100_affine_paired_laws_v1 FAIL; sustained coverage FAIL, 0/5 terminal joint passes. Independent100k clean/live holdout: centre RMS 0.229900289sigma, covariance bias -0.105827929, radialKS 0.039679289, massTV 0.080790000, precision 0.97504. All G/D/prior updates complete, finite/hooks checks PASS, unintended RNG deviations0, independent receipt/artifact/evaluator audit PASS. Unique paid 75.190272702981s. No ordinary qualification or promotion; all unknown cells preserved.

**Next:** Stop these exact revisions and this selected registration. Before another GPU study, perform a bounded zero-training saved-state audit of critic radial and centering gradients against per-mode moments. Only a supported substantive hypothesis warrants a new frozen registration. Preserve A2 and sigma .025; no coefficient, cap, width, amplitude or seed sweep, best-checkpoint/EMA substitution, 14k continuation or automatic matrix expansion.

[Record and receipt provenance](records/readout-d5a99d9470252f695a717dfc.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-d84b69a82debc8d2678cf004

**Scope:** research_diagnostic; scientific; revision `589de06395db614a222a31d5969b521a8ce6560ca155439b937772898fd6f2d7`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.626; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-d84b69a82debc8d2678cf004.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2--f78d5d603be1611025389c4ca0f7f485f3d2caa0a25fb6bc38b7c96da3509a3e · readout-da2ac34b1675880ab8ace248

**Scope:** current; scientific; revision `ecd7497dc47ac901328a0e28a32d2657cf2ce0601db5690c629b1f33a6626d45`.

Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.

**Observed:** {'FAIL': 1}; wall seconds 8.29; mechanism `floor_constant`.



This exact global configuration failed the first two_pole movement gate. Tier1 is0/5; word and25remainingcells are UNKNOWN. No configured standard or default adoption.

**Next:** Bounded global search concluded. Retain the exact historical family incumbent as unqualified evidence. No further paid stage is authorized; require a separate bounded global proposal before any continuation.

[Record and receipt provenance](records/readout-da2ac34b1675880ab8ace248.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-r1r2-v1 · readout-da78d1d541082f723b706f2c

**Scope:** research_diagnostic; scientific; revision `ec161216b6da8e296c0ccd3125ade1e37245f3cea1f3f24f261935c6951cf71b`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 12}; wall seconds 119.889; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-r1r2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-da78d1d541082f723b706f2c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-throughput-v1 · readout-dbb540fd1a0d83052f31aef8

**Scope:** research_diagnostic; scientific; revision `5519ffc77cf0cfd46928ca25fbbf29efb9a642d8da333482419f4c9af3a4300d`.

Measure update throughput and preserve the sustained stripe PASS on unchanged small-image task contracts. Retain the held bars outcome.

**Observed:** {'FAIL': 1, 'PASS': 1}; wall seconds 30.959; mechanism `structural`.

img_bars4: modes=2, hq=0.96875; img_stripes2: modes=2, hq=1

Original small stripes PASS and bars FAIL; all 600 updates and 24 observations complete.

**Next:** Retain the global batched backend as an opt-in; keep the narrow bars failure visible.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-dbb540fd1a0d83052f31aef8.json)

### recipe-prior-refactor-k3p-v1 · readout-dbd37db03b844fa11e42af29

**Scope:** research_diagnostic; scientific; revision `7f72d6fed367b3719b8f73d283ee6290b78f987509f573ad7860d2d3a46a576d`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.943; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-dbd37db03b844fa11e42af29.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2--7728cf93188b9fe122536244667789941da52d435cfb0a3ae42b90db200d7595 · readout-dc31730b85dde3cf85253359

**Scope:** current; scientific; revision `059128df90e1bcb0af7f4596f5f9dd034686b171c24177f080a2b971e0f79fe2`.

Under the unchanged fixed toy-host movement/convergence gates, larger constant learning rates and paired R1/R2 gamma schedules may improve the failed Modern GAN training adaptation.

**Observed:** {'FAIL': 1}; wall seconds 9.516; mechanism `floor_constant`.



Bounded R1/R2 Modern GAN toy-host configuration {'lr': 0.00425, 'reg_coeff': 1.0, 'reg_coeff_end': 0.1} under the fixed source/protocol/runtime produced [{'task': 'two_pole', 'gate_status': 'FAIL', 'metrics': {'grad_med': 0.2411961704492569, 'mean_abs': 0.14577166736125946}}]. Ordinary prerequisite stopping retained every unmeasured requirement.

**Next:** Stop this exact failed configuration; review the provisional convergence screen before another parameter study. No downstream diagnostic, seed repeat or default adoption follows.

[Evidence](../../reports/forge/technique-receipts/b686547339464e94a60fd692675f1545.json) · [Record](records/readout-dc31730b85dde3cf85253359.json)

### bispf-optimized-quintic-v1 · readout-dc5a72705708c9da1685c19d

**Scope:** research_diagnostic; scientific; revision `55d2bb6b0fcf3af305945e5bd99f9aadef2d5d0899bfb579ff10b7a8210e5ab0`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 63.526; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-dc5a72705708c9da1685c19d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-release07-gan-v3-v1 · readout-dd6ed9cfc7ac77bebc9b2bda

**Scope:** research_diagnostic; scientific; revision `fa050d0eea980c0d8d79a813ae2f45149e1cd5c9046d102dffe0f648923926b0`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.942; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-release07-gan-v3-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-dd6ed9cfc7ac77bebc9b2bda.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p-global-repair-v1 · readout-dd6ff59094d1d766c5b0fe2b

**Scope:** research_diagnostic; scientific; revision `bf98d947a19472491d60388e43c6af2039f0f98fb874a6498614febba5fdfbe6`.

The exact repaired global recipe acquires the five-word inverse task and transfers without task-specific optimizer tuning across every eligible required discriminator-stability task under ordinary prerequisites.

**Observed:** {'FAIL': 2}; wall seconds 18.457; mechanism `structural`.



Both800-update diagnostics FAIL movement: original-schedule .0896976665 and stretched-schedule .1111024097 versus .3. More steps did not rescue this existing global word-positive recipe.

**Next:** See reports/forge/k3p-two-pole-horizon-v1/README.md and saved force analysis. Retain the main gate and family selection. Investigate existing global critic/particle balance before any new separately bounded search; no automatic follow-on work.

[Record and receipt provenance](records/readout-dd6ff59094d1d766c5b0fe2b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-e2a893a94b380b21b52fde5c

**Scope:** research_diagnostic; scientific; revision `bfad40e7a33bb7b18a761191623a6e2cf66e4e39ec260b85d3a76369912fa5fd`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.742; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-e2a893a94b380b21b52fde5c.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-ddgan-energy-annealed-v2 · readout-e2e0ab5b4a57b6aae957eec4

**Scope:** research_diagnostic; scientific; revision `ae4202e9b9ea41b2007a5b37e81e695839718df0206bba202660de2124b8c8ef`.

A globally smaller normalized learning rate with positive cosine terminal floors stabilizes conditional-energy DDGAN calibration while preserving its stochastic distribution objective and genuine adversarial game.

**Observed:** {'FAIL': 2}; wall seconds 1228.839; mechanism `floor_constant`.



Both full-budget DDGAN tasks remain FAIL. Gaussian passes all5 final marginal checks but retains fixed t1_q0 conditional errors. Bimodal improves marginal/mode means but fails local spread/HQ and conditional terminal gates.

**Next:** Execute the already registered final one-field objective-geometry diagnostic: constraint_geometry_mode direction_blend to none, preserving the same summed GAN/energy/SBNP objectives, coefficients, rates, scientific source, all gates, fixed queries and complete budgets. No ordinary qualification or automatic continuation.

[Record and receipt provenance](records/readout-e2e0ab5b4a57b6aae957eec4.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### ka2--4496b859e066d7a4b6279f2d34fbd58d083e0f18c2a22ade5ba12c964d01a7ca · readout-e3ab30a3f7ae4a6debe66e5d

**Scope:** current; scientific; revision `0cb6ee2042006b122ef7c2c91b9ecdc74b703aa8ee51ac3e36a5164abde58a98`.

Increasing the global network/base rate while retaining the nominal absolute latent-table prior base rate may restore movement within the unchanged 80-update screen. Every complete configuration is then evaluated through all ordinary Tier 3 prerequisites; later failures remain part of this whole-candidate result.

**Observed:** {'FAIL': 1}; wall seconds 5.995; mechanism `floor_constant`.



This exact global configuration failed the first two_pole movement gate. Tier1 is0/5; word and25remainingcells are UNKNOWN. No configured standard or default adoption.

**Next:** Bounded global search concluded. Retain the exact historical family incumbent as unqualified evidence. No further paid stage is authorized; require a separate bounded global proposal before any continuation.

[Record and receipt provenance](records/readout-e3ab30a3f7ae4a6debe66e5d.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-throughput-v1 · readout-e5cf5311e83c25cc0b333270

**Scope:** research_diagnostic; scientific; revision `5519ffc77cf0cfd46928ca25fbbf29efb9a642d8da333482419f4c9af3a4300d`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact declared global recipe.

**Observed:** {'PASS': 6}; wall seconds 60.995; mechanism `structural`.



All six finite-update DCGAN performance tasks PASS. Observed speedups versus matched quintic controls range 1.02×–2.15×, geometric mean 1.66×;256 px 2.15× and batch 64 2.14× meet the target.

**Next:** Offer the two-field opt-in recipe; profile remaining 32/64 px overhead before another bounded implementation change. Keep defaults and existing selection pins unchanged.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-e5cf5311e83c25cc0b333270.json)

### r1r2-repair-r1only-20261022 · readout-e610d5af4bac226eabe47dcb

**Scope:** research_diagnostic; scientific; revision `704c803e0c8bf425e1c0cd673b77e0332278cd8163ca9058a95c50a2f00fbdab`.

This exact source-bound global recipe can retain its own earliest independently confirmed word acquisition through every original scheduled primary and independent confirmation check, including the exact restored state, for4000 further updates. The original word_hold evaluator and all25 paired checks are authoritative; endpoint token probability alone cannot establish retention. This diagnostic changes scheduler placement only and grants no ordinary qualification.

**Observed:** {'PASS': 2}; wall seconds 141.642; mechanism `structural`.

five_word_joint_hold: modes=5, mass_tv=0.03457; five_word_joint_smoke: modes=5, mass_tv=0.018945

Own-checkpoint acquisition and full4000-update hold PASS: all25 scheduled/restored-state checks confirmed, parent update1667, endpoint minimum token probability0.9974690675735474. No ordinary qualification credit. The NS/AMSGrad reference already passes retention; explicit reconstruction improves confidence margin but is not necessary for these gates.

**Next:** Close the finite three-role campaign. Keep R2; do not promote the failed global repairs. Preserve all ordinary failures and archived source identities. Require a demonstrated density or conditional-field prerequisite before another candidate.

[Evidence](../../reports/forge/r1r2-repair-20261022/README.md) · [Record](records/readout-e610d5af4bac226eabe47dcb.json)

### recipe-prior-refactor-k3p-v1 · readout-e6cf768866c160ba1338f662

**Scope:** current; scientific; revision `e3680ce98264605af83a69b7600141a633a7c42c93d95c50bc4ce762bc2c0acc`.

Ordinary re-qualification of the unchanged selected k3p recipe (recipe-prior-refactor-k3p-v1) on current develop source 2e8a97a0aa84 under protocol seed 0 on NVIDIA RTX A6000 records current-source discriminator_stability and shared Tier 1 goal-view verdicts.

**Observed:** {'FAIL': 3, 'PASS': 9}; wall seconds 412.715; mechanism `structural`.

ring16_acquisition: modes=16, hq=0.77905, mass_tv=0.062988

Completed discriminator_stability seed-0 CUDA execution for recipe-prior-refactor-k3p-v1 (FAIL 3, PASS 9). Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact.

**Next:** Retain source-bound readout and compiled Forge reports.

[Evidence](../../) · [Record](records/readout-e6cf768866c160ba1338f662.json)

### recipe-prior-refactor-ka2-v1 · readout-e6e872f73024fd6c0b2b2d42

**Scope:** research_diagnostic; scientific; revision `d271e37be7bfbc9c110b7acecd2d78316eaa1360d4af9db352b1fe26cb1e46a1`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 67.169; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/ka2-dcgan_performance_b64-0dcd26728de3eb07.json) · [Record](records/readout-e6e872f73024fd6c0b2b2d42.json)

### recipe-prior-refactor-release07-gan-v3-v1 · readout-e838608cfd803f50b58f4e21

**Scope:** research_diagnostic; scientific; revision `9c4c532b8c3773b5f17b7b2ec3f5729d4920bd9b6e409609b7da88059ea3fa2f`.

Evaluate whether unchanged selected release07-gan-v3 recipe (recipe-prior-refactor-release07-gan-v3-v1) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 55.848; mechanism `structural`.



Selected release07-gan-v3 recipe passes interp_img_patch_shift (5/5 terminal checks: interp_rmse=0.0477, interp_peak=0.8727) and passes every gate on the final 2 checks of interp_saddle_barycentric (step 800 barycenter_surface_mae=0.0587, barycenter_hq=0.8056), while cutting chords on sparse_arc (endpoint_hq=1.000, interp_hq=0.6522).

**Next:** Test whether faster prior transport warmup allows release07-gan-v3 to reach its passing saddle_barycentric regime across all 5 terminal checks.

[Record and receipt provenance](records/readout-e838608cfd803f50b58f4e21.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-e8818ca1024f2d2972606913

**Scope:** current; scientific; revision `6469950271908090f8608849192d212372abf993333a4207a024bfca7ac39d4c`.

Ordinary re-qualification of the unchanged selected ka2 recipe on the current develop source reproduces its recorded discriminator_stability outcomes.

**Observed:** {'BLOCKED': 1, 'FAIL': 2, 'INCOMPLETE': 4, 'PASS': 5}; wall seconds 268.104; mechanism `structural`.

ae_gan_hold: recon_mse=0.0036862

Ordinary discriminator_stability revision-14 re-qualification reproduces all 7 core Tier 1 verdicts (5 PASS, ring16_acquisition FAIL, clockfree_audit_measurement_v1 FAIL) and stops at Tier 1; revision-9 qualification retained.

**Next:** Retain revision-9 qualification in technique-inventory; address ring16_acquisition and clockfree_audit_measurement_v1 in family follow-up.

[Record and receipt provenance](records/readout-e8818ca1024f2d2972606913.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-throughput-v1 · readout-e96805cfab19f80314e3ec20

**Scope:** current; scientific; revision `5519ffc77cf0cfd46928ca25fbbf29efb9a642d8da333482419f4c9af3a4300d`.

Preserve every counted and previously passing ordinary BISPF gate with one global throughput recipe. Retain every held failure.

**Observed:** {'FAIL': 11, 'PASS': 17}; wall seconds 2463.782; mechanism `structural`.

ae_gan_hold: recon_mse=0.0036687

The unchanged ordinary revision 12 tier completes 28 tasks with 17 PASS and 11 FAIL. All 17 measured incumbent ordinary passes remain; four restored required failures block ordinary Tier 3. The separately scoped endurance study supplies two further preserved numerical passes, not qualification.

**Next:** Retain the opt-in throughput recipe and existing selected/default recipe; do not claim revision 12 Tier 3 qualification or tune the unchanged failed gates as part of this latency study.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-e96805cfab19f80314e3ec20.json)

### atlas-particle-normal-v2 · readout-e97f5dbf54d7493604c4689f

**Scope:** research_diagnostic; scientific; revision `25cbc5b8c787807ef64f0bb8f8e65eea25e3e13940f6fc90b733063baab912fc`.

De-clocking Atlas's critic guard (d_guard_ratio=0.0, d_guard_min_steps=0), restoring LossSurprise equilibrium separation (Z_THRESH=3.5, Z_CALM=1.5), and binding clean/live public_components word_joint and model task snapshots allows atlas-particle-normal-v2 to pass clock-free parity, mode_hold, joint word acquisition/hold, and AE reconstruction on the 48-task full suite under protocol seed 0.

**Observed:** {'FAIL': 16, 'PASS': 22}; wall seconds 3254.541; mechanism `structural`.

staggered100: precision=0.99314, center_rms_sigma=0.13671, mass_tv=0.02165, radial_ks=0.15584; ae_gan_hold: recon_mse=0.0099752

Under protocol seed 0 on prior-ablation-full-suite-v1 (NVIDIA RTX A6000), atlas-particle-normal-v2 achieves 22 PASS, 16 FAIL, 0 INCOMPLETE across 38 executed tasks (6 prerequisite-gated, 4 unimplemented-host blocked). Prediction on clockfree_audit observed (parity_comparisons=4 >= 4, 0 unexplained clock dependencies) and falsifier on gaussian1d_smoke avoided (mean_error_sigma=0.0626 <= 0.20). Clean training (output_noise_mode='fixed', output_noise_std=0.0), de-clocked critic guard (d_guard_ratio=0.0, d_guard_min_steps=0), restored LossSurprise equilibrium hysteresis (Z_THRESH=3.5, Z_CALM=1.5), and public_components/task-snapshot Forge bindings flip 5 tasks to PASS (clockfree_audit_measurement_v1, clockfree_audit, mode_hold, img_intensity2, model_ae_reconstruction) with 0 regressions and 0 INCOMPLETE errors.

**Next:** Introduce a clock-free state-gated critic payoff/ratio guard (activating only after initial critic discrimination settles rather than at step 0 or fixed step 200) to recover five_word_joint_smoke/hold without regressing mode_hold or img_intensity2, and reduce clean per-mode covariance trace deficit (cov_trace_bias ~ -0.30 on 100-mode grids and ring16_acquisition) via local Jacobian spread or lower stationary R1/R2 damping.

[Record and receipt provenance](records/readout-e97f5dbf54d7493604c4689f.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-k3p-v1 · readout-ea9a34c10e46857f904487a0

**Scope:** research_diagnostic; scientific; revision `14d360c8d0b026ef6bc21b4e98e560a15538ec8d10eda49770d1f96f55b90ace`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 66.915; mechanism `structural`.



All six declared DCGAN mock-data performance workloads completed their finite public-trainer execution budgets (3 warmup + 10 measured updates, seed 0). Diagnostic execution evidence only; no image-quality, qualification or default-adoption claim.

**Next:** Use the published scaling measurements to budget larger-image pilots; restore verifiable contention telemetry before paired timing claims. Profile complete-update components before changing recipes. Keep the default unchanged and require separate quality evidence for adoption.

[Evidence](../../reports/forge/scaling-receipts/dcgan-performance-cloud-v2/k3p-dcgan_performance_b64-d323d8672d36f9c0.json) · [Record](records/readout-ea9a34c10e46857f904487a0.json)

### bispf-optimized-quintic-v1 · readout-eb425a0e7fa0639a1081f369

**Scope:** research_diagnostic; scientific; revision `51757701e512bd8c20311cc071e481418dc1a16e73eb8d153f79b30d78b994c6`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 64.98; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for bispf-optimized-quintic-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-eb425a0e7fa0639a1081f369.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### k3p · readout-eeec25002d17191c315f126b

**Scope:** calibration_diagnostic; scientific; revision `6385ef80463dca6a3c1c1f94dd3269c8e1db3702a8d236c1bd59d69787e06325`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 1}; wall seconds 91.24; mechanism `structural`.

grid100_affine_paired_laws_v1: precision=0.99517, center_rms_sigma=0.089885, mass_tv=0.032, radial_ks=0.12001

Preregistered full7000-update grid100_affine_paired_laws_v1 FAIL; sustained coverage PASS, 0/5 terminal joint passes. Independent100k clean/live holdout: centre RMS 0.089885161sigma, covariance bias -0.265593703, radialKS 0.120006288, massTV 0.032000000, precision 0.99517. All G/D/prior updates complete, finite/hooks checks PASS, unintended RNG deviations0, independent receipt/artifact/evaluator audit PASS. Unique paid 91.239885760006s. No ordinary qualification or promotion; all unknown cells preserved.

**Next:** Stop these exact revisions and this selected registration. Before another GPU study, perform a bounded zero-training saved-state audit of critic radial and centering gradients against per-mode moments. Only a supported substantive hypothesis warrants a new frozen registration. Preserve A2 and sigma .025; no coefficient, cap, width, amplitude or seed sweep, best-checkpoint/EMA substitution, 14k continuation or automatic matrix expansion.

[Record and receipt provenance](records/readout-eeec25002d17191c315f126b.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### atlas-particle-normal-v3 · readout-efb8bacf93e19fc2b0d7f7bf

**Scope:** research_diagnostic; scientific; revision `4e3279bde6957fc95c5ba573328f9b9379a0c737ad5a7f06f637105b74e7ddf0`.

Raising birth_death_cells from 128 to 256 eliminates under-populated 2-particle-per-cell FeatureCellBirthDeath resampling noise on N=256 vector transfer tasks while refining N=20000 cell resolution, and raising NormalNoiseManipulator sigma from 0.005 to 0.0065 sustains rare-mass and multi-scale intra-mode covariance, flipping both vector_unequal_mass and vector_unequal_width to PASS (6/6 vector tasks) and raising full-suite score to >= 23 PASS / 0 INCOMPLETE on prior-ablation-full-suite-v1 under protocol seed 0.

**Observed:** {'FAIL': 12, 'PASS': 23}; wall seconds 3285.431; mechanism `structural`.

staggered100: precision=0.98711, center_rms_sigma=0.15778, mass_tv=0.02468, radial_ks=0.077528; five_word_joint_smoke: modes=0, mass_tv=1; vector_two_broad: hq=0.98657, mass_tv=0.016602

Raising birth_death_cells from 128 to 256 (eliminating noisy 2-particle-per-cell FeatureCellBirthDeath resampling on N=256 vector transfer tasks while refining N=20000 cell resolution) and NormalNoiseManipulator sigma from 0.005 to 0.0065 deterministically achieves 23 PASS, 12 FAIL, 0 INCOMPLETE across 35 executed tasks (13 prerequisite-gated tasks deferred) on prior-ablation-full-suite-v1 under protocol seed 0 (NVIDIA RTX A6000). Observes predicted vector_unequal_mass component_covariance_error=0.30664 <= 0.85 (down from 1.2828 in v2) and flips both vector_unequal_mass and vector_unequal_width (component_covariance_error=0.34328 <= 0.85, down from 2.3235 in v2) to PASS for 6/6 vector transfer passes alongside 4/4 image passes, 2/2 clockfree audit passes, and 2/2 VAE passes, while avoiding the gaussian1d_smoke falsifier (mean_error_sigma=0.04435 <= 0.2).

**Next:** Investigate clean-served intra-mode covariance expansion on N=20000 100-mode tasks (grid100/staggered100/rotated100) without repulsive prior penalties that perturb zero_tanh_smoke or enumerated 32-particle image tables, and address joint BiGAN critic balance on five_word_joint_smoke.

[Record and receipt provenance](records/readout-efb8bacf93e19fc2b0d7f7bf.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### recipe-prior-refactor-ka2-v1 · readout-efc9b823a6b391bc25754f9a

**Scope:** current; scientific; revision `344ea72637958be757f2e3ac043897a54cfa68667e05bf1a116397effc02a93e`.

Ordinary re-qualification of the unchanged selected ka2 recipe (recipe-prior-refactor-ka2-v1) on current develop source 2e8a97a0aa84 under protocol seed 0 on NVIDIA RTX A6000 records current-source discriminator_stability and shared Tier 1 goal-view verdicts.

**Observed:** {'FAIL': 3, 'PASS': 9}; wall seconds 370.958; mechanism `structural`.

ring16_acquisition: modes=16, hq=0.76392, mass_tv=0.059082; model_ae_reconstruction: modes=4, precision=0.061279, mass_tv=0.088379

Completed discriminator_stability seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (FAIL 3, PASS 9). Ordinary current-source seed-0 CUDA measurement recorded; archived historical pins remain intact.

**Next:** Retain source-bound readout and compiled Forge reports.

[Evidence](../../) · [Record](records/readout-efc9b823a6b391bc25754f9a.json)

### r1r2--f398f6ea70a2d101c520a1a01db90ea968bbb16736e6cec7ffc887e1dd7e241a · readout-f0dadd320f7db48cbf5d7b20

**Scope:** current; scientific; revision `158a5e917e91a158e45de1e37434b11e2469a8e39fbca5b99edac6069c7764aa`.

Under the unchanged fixed toy-host movement/convergence gates, larger constant learning rates and paired R1/R2 gamma schedules may improve the failed Modern GAN training adaptation.

**Observed:** {'FAIL': 1}; wall seconds 5.513; mechanism `floor_constant`.



Bounded R1/R2 Modern GAN toy-host configuration {'lr': 0.0085, 'reg_coeff': 0.1, 'reg_coeff_end': 0.01} under the fixed source/protocol/runtime produced [{'task': 'two_pole', 'gate_status': 'FAIL', 'metrics': {'grad_med': 1.0202852487564087, 'mean_abs': 1.0062839984893799}}]. Ordinary prerequisite stopping retained every unmeasured requirement.

**Next:** Stop this exact failed configuration; review the provisional convergence screen before another parameter study. No downstream diagnostic, seed repeat or default adoption follows.

[Evidence](../../reports/forge/technique-receipts/2a719ec706924f0ab120961cd58f7582.json) · [Record](records/readout-f0dadd320f7db48cbf5d7b20.json)

### recipe-prior-refactor-ka2-v1 · readout-f49e05e15c9423e6a7e9d1b0

**Scope:** research_diagnostic; scientific; revision `599e3b4f3bcbc4b5667a35b1fa60d7334ddb5b3b924d88e514d11eee930db237`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 58.069; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-ka2-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-f49e05e15c9423e6a7e9d1b0.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-f569a1b3b7d32568c334955a

**Scope:** research_diagnostic; scientific; revision `b4bddd6529b2782b15a82e342c4c7b275ebf2962d30479c852134d7920e3a387`.

Evaluate whether unchanged selected bispf recipe (bispf-optimized-quintic-v1) with zero-sigma discrete ParticlePrior (no MoG smoothing) learns smooth, non-snapping latent interpolants across unseen 1D latent k-NN chords (interp_sparse_arc, interp_img_patch_shift) and 2D simplex barycenters (interp_saddle_barycentric).

**Observed:** {'FAIL': 3}; wall seconds 92.274; mechanism `structural`.



Selected bispf recipe fails all 3 zero-sigma ParticlePrior interpolation tasks (0/3 PASS), reaching barycenter_surface_mae=0.0712 <= 0.09 and barycenter_hq=0.7286 on interp_saddle_barycentric while cutting chords on sparse_arc (interp_hq=0.3788) and dimming midpoints on img_patch_shift (interp_peak=0.6189).

**Next:** Analyze why quintic spectral normalization preserves 2D saddle curvature better than 1D arc chords without MoG kernel overlap.

[Record and receipt provenance](records/readout-f569a1b3b7d32568c334955a.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### r1r2--b58a087cabab6b43997ec233416294cfc2786bab59026c07a1a4f2565970f5ff · readout-f61373e09376e33fc4c501c8

**Scope:** current; scientific; revision `ff171334c936fd2a0bc7ad4e4beb25ba6582bc35e3baf2042db7345738705493`.

Under the unchanged fixed toy-host movement/convergence gates, larger constant learning rates and paired R1/R2 gamma schedules may improve the failed Modern GAN training adaptation.

**Observed:** {'FAIL': 1}; wall seconds 5.456; mechanism `floor_constant`.



Bounded R1/R2 Modern GAN toy-host configuration {'lr': 0.00425, 'reg_coeff': 0.1, 'reg_coeff_end': 0.01} under the fixed source/protocol/runtime produced [{'task': 'two_pole', 'gate_status': 'FAIL', 'metrics': {'grad_med': 0.6937375068664551, 'mean_abs': 0.37995293736457825}}]. Ordinary prerequisite stopping retained every unmeasured requirement.

**Next:** Stop this exact failed configuration; review the provisional convergence screen before another parameter study. No downstream diagnostic, seed repeat or default adoption follows.

[Evidence](../../reports/forge/technique-receipts/8ff64e32bf294e5597d8deb036527075.json) · [Record](records/readout-f61373e09376e33fc4c501c8.json)

### empf-v7 · readout-f65f1b132b323a6a5b40bef5

**Scope:** research_diagnostic; scientific; revision `4989a6ff9820637f735ce48ffadca09683266cf4fd65086d7328d35ad78ee441`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 59.501; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for empf-v7 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-f65f1b132b323a6a5b40bef5.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bispf-optimized-quintic-v1 · readout-f7a1657c1890cbb69c8d89f4

**Scope:** research_diagnostic; scientific; revision `fbc1bf4bda35c195efe7830c359d1851febdbca85b4257b4415d66187ed3d248`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact declared global recipe.

**Observed:** {'PASS': 6}; wall seconds 64.729; mechanism `structural`.



All six current-source quintic DCGAN controls complete their frozen finite-update budgets with finite states.

**Next:** Archive the complete matched matrix and preserve the current selected/default recipe; no unchanged reruns.

[Evidence](../../reports/forge/bispf-throughput/README.md) · [Record](records/readout-f7a1657c1890cbb69c8d89f4.json)

### recipe-prior-refactor-bcap-pure-v1 · readout-f878dc3aade4f1e6cc109a58

**Scope:** research_diagnostic; scientific; revision `77765918f0b8e49975abeef0068a12cef5c452594aa5d8f94003f0eaedfd3f52`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 162.852; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for recipe-prior-refactor-bcap-pure-v1 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-f878dc3aade4f1e6cc109a58.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### bcap-dualnorm--8db70e3cb9fd3da9b5cc6a117731e8572cba837d64e7ee721d012d3157c9a3fe · readout-fa5fada954b7b255ec86d748

**Scope:** current; scientific; revision `72d9237558743f721c8636feb4642e4790c0f54ac1b139fdd0948daf096dfbf8`.

The single selected constant-rate BCAP/DualNorm smoothing=1e-5 recipe that acquired all six Tier 1 tasks can retain live quality and continued learning across the unchanged 21 required Tier 2 tasks. This is a frozen-recipe qualification stage, with no further tuning, seeds or learning-rate annealing.

**Observed:** {'FAIL': 11, 'INCOMPLETE': 4, 'PASS': 13}; wall seconds 2005.605; mechanism `floor_constant`.



Frozen BCAP smoothing=1e-5 retains 6/6 required Tier 1 passes; Tier 2 records {'FAIL': 11, 'PASS': 6, 'INCOMPLETE': 4}. Gaussian retains only 1/72 stationary checks and misses shifted reacquisition/hold. Words retain generation but lose inverse reconstruction; four image hosts reject convolution tensors before training. No learning-rate annealing, scientific retries, Tier 1 reruns, Tier 3 or default adoption. The frozen hypothesis signature misnames cdf_ks as ks; preserve its incomplete decision and report the actual scalar separately.

**Next:** Keep the Tier 1 selection and these complete Tier 2 failures. Inspect saved retention and inverse-map trajectories; declare any smaller constant global rates or stronger fixed smoothing as a new bounded comparison. Convolution support requires a separately declared optimizer adaptation. Do not spend on Tier 3.

[Evidence](../../reports/forge/bcap-tier2/readout.json) · [Record](records/readout-fa5fada954b7b255ec86d748.json)

### k3p · readout-faf4999394106140c8a2052e

**Scope:** current; scientific; revision `2a97b74e933cc5199847bc89b917f6643d9fd82f8deb2a8490d29b63fba718ce`.

The shipped K3P mechanism is a reference for new learned-MoG experiments; historical cloud evidence does not qualify this MoG conversion.

**Observed:** {'FAIL': 1}; wall seconds 5.581; mechanism `structural`.



This technique failed the full 80-update two_pole smoke gate: {'grad_med': 0.0637243241071701, 'mean_abs': 0.10652091354131699}. One completed attempt cost 5.581290134 paid seconds. The remaining 2 smoke, all 19 quality and both endurance requirements remain unknown; no execution error or downstream spending.

**Next:** Stop this exact failed ordinary cohort; preserve the negative and unknown denominator. Do not rerun it, vary seeds, fill downstream gates or promote it. Calibration requires a supported bounded design, not relaxed thresholds.

[Evidence](../../reports/forge/technique-receipts/3e1d8ca0f47a4954b41119cbe605548e.json) · [Record](records/readout-faf4999394106140c8a2052e.json)

### atlas-particle-normal-v4 · readout-ff53c8e071687458a4c51315

**Scope:** research_diagnostic; scientific; revision `efebaebc63fbb814b384b2009aedaf06465137a061cb9fe618b83c9a326f82dd`.

Measure complete public GANTrainer update time and memory as image resolution, batch and width grow, preserving the exact selected global recipe.

**Observed:** {'PASS': 6}; wall seconds 62.685; mechanism `structural`.



Completed dcgan_performance seed-0 CUDA execution for atlas-particle-normal-v4 (PASS 6). Diagnostic performance/research scope only; no qualification credit.

**Next:** Retain source-bound readout and compiled Forge reports.

[Record and receipt provenance](records/readout-ff53c8e071687458a4c51315.json) · [Original artifact retrieval](../../docs/forge-artifact-resolver.md). Original receipt availability is separate from this record.

### ring16-duration-20261011 · ring16-duration-20261011-readout

**Scope:** task_only_diagnostic; scientific; revision `257f8c355e756a3d615c369ef3774220951ff6bf121b60a4392a39b98a50d4de`.

More unchanged training may repair the three selected ring density failures beyond 1,600 updates.

**Observed:** {'FAIL': 3}; wall seconds 137.861; mechanism `execution_budget_only`.

ring16_acquisition__r1r2__duration6400: modes=16, hq=0.94775, mass_tv=0.051514; ring16_acquisition__ka2__duration6400: modes=16, hq=0.87598, mass_tv=0.064697; ring16_acquisition__k3p__duration6400: modes=16, hq=0.89648, mass_tv=0.070312

Three exact archived public-API CUDA continuations add 14,400 updates without replaying prefixes. All three fail the terminal gate at 6,400 updates. R1/R2 first reaches five consecutive passes at update 5,217 and its longest streak is 19 checks. It ends with only three terminal passes because minimum eigenvalue spread fails at update 6,350 (0.13328 < 0.15), despite a fully passing endpoint: HQ 0.947754 and covariance error 0.525637. KA2 and K3P have no full passing observations and finish with HQ 0.875977/0.896484 and covariance error 1.783269/1.599982. More unchanged updates improve all three recipes, but duration alone does not establish reliable terminal quality. The 400-update schedule stays fixed, so this study does not isolate schedule decay as the cause. Exact restores, cap-only extensions, final optimizer counts, complete consumed RNG checkpoints, matching continuation target batches and saved-sample checks pass. Original 1,600-update FAIL grades and the current leaderboard remain unchanged.

**Next:** Stop this bounded duration round. A separately declared schedule-horizon comparison at a fixed 6,400-update cap is the next useful option for KA2/K3P. R1/R2 also needs local spread retention assessed. Preserve recipes, prior, target, architecture, seed 0, cadence and full gates; reuse the saved longer-budget baseline. No automatic follow-up, seed study, qualification or default adoption.

[Evidence](../../reports/forge/ring16-duration-20261011/results.json) · [Record](records/ring16-duration-20261011-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · ring16-failure-saved-diagnosis-v1

**Scope:** saved_output_and_checkpoint_diagnosis; family_context; revision `e90fc925c996dccff3a054913a040cd4b3b76023972294062139fe5815d7db50`.

Explain why the adopted 1600-update Ring16 conditions retain an archived continuation PASS but fail current uninterrupted training before proposing another solution.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `saved_evidence_analysis`.



The archived restored 1600-update PASS and current uninterrupted 1600-update FAIL share recipe, prior, initializer, 91 scientific files and all 14 final named RNG states. All 24 prefix observations through 400 match exactly; the first differing saved observation is 417. All 96 v3/v4 samples and metrics match. Current covariance error 2.22027 fails with zero full passes: 21 of 4096 outputs exceed 20 target sigmas, all assigned to component 11, which contributes 82.3% of the summed covariance errors. The other 15 components average 0.41875. The current roster records 22 numerical FAIL and one API BLOCKED. The exact first differing optimizer update and causal mechanism remain unresolved; original grades are preserved.

**Next:** Declare a bounded CUDA diagnostic comparing continuation of a shared live prefix with restoration of its checkpoint before tuning. Then attribute far outputs to sampled prior rows and local generator mappings. A pass-once rule cannot repair zero full passing observations. This report adds no training, qualification or policy change.

[Evidence](../../reports/forge/ring16-failure/results.json) · [Record](records/ring16-failure-saved-diagnosis-v1.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · ring16-restart-diagnostic-v1

**Scope:** cuda_checkpoint_boundary_reproduction; family_context; revision `e90fc925c996dccff3a054913a040cd4b3b76023972294062139fe5815d7db50`.

Reproduce what actually changes when restoring Ring16 at400 versus continuing the live objects, with seed0, public initialization, recipe and draws fixed.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `numerical_sensitivity_diagnostic`.



Restoring the fresh 400-update state and the original archived state each reproduces all 96 historical passing sample tensors exactly (covariance .514315, six terminal passes). Live continuation reproduces all 96 current failing covariance/HQ observations (final covariance 2.220268), but retains its postexecution metadata-error INCOMPLETE receipt and no qualification. At update 401, weights, real/latent inputs and the first six forwards agree; critic weight gradients differ by 1e-9 to 1e-8. A 1.03e-7 relative hidden-gradient change becomes a .252 relative polar-direction change because near-null singular directions receive unit weight. Named and ambient RNG states, batches and sampled indices match all 16 traced updates; four controls match the restored path exactly. Restoring live gradient buffers and module flags does not alter that result. No seed variation, scheduled randomization or subbatch repeats were tested.

**Next:** Investigate damping or truncating almost-null singular directions under a separately declared global trainer delta, constant rates and unchanged full gates. Exact backward accumulation mechanism remains unresolved. Do not promote periodic reloads from this one fixed-state reproduction.

[Evidence](../../reports/forge/ring16-failure/reproduction-results.json) · [Record](records/ring16-restart-diagnostic-v1.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · ring16-runtime-and-interventions-v1

**Scope:** cuda_runtime_and_weak_direction_diagnostics; family_context; revision `e90fc925c996dccff3a054913a040cd4b3b76023972294062139fe5815d7db50`.

Isolate runtime rounding after Ring16 reload and test fixed weak-direction interventions under the same public initialization, prior, batches, constant rates and full distribution bounds.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `numerical_sensitivity_diagnostic`.



Four CUDA controls exactly reproduce historical ordinary critic gradients. Serializing autograd only during update401 makes complete live/restored contexts identical but chooses a third trajectory; actual engine execution order and serialized1600 quality remain unmeasured. Identical topology but 1054/7503 changed sequence relations, versus zero for serialized controls, supports thread-local autograd counter history and accumulation ordering. No lossy checkpoint dtype conversion found. Continuous truncation confirms acquisition at684 (47terminal passes), continuous tiny-gradient noise at817 (45), and one-time sign flips at1300 (19). Truncation fails covariance at717/734/750/817 and noise at850 after acquisition, so never-leaves-goal retention is not established. Damping/noise at401 pass terminal quality but fail their one first-pass confirmation; continuous damping/sign flips have zero full passes. All original receipt statuses and gates are preserved.

**Next:** Prioritize deterministic continuous truncation and retain weak-gradient noise as a second supported candidate for a separately frozen whole-configuration Tier1 comparison. Exact reload attribution needs scoped graph-construction/scheduling controls or executed-node order capture. Run Tier2 retention only after required ordinary gates pass. Stop completed every-step damping/sign-flip negatives; no seed study, automatic continuation, default adoption or qualification follows.

[Evidence](../../reports/forge/ring16-followup/results.json) · [Record](records/ring16-runtime-and-interventions-v1.json)

### ring16-schedule-20261011 · ring16-schedule-20261011-readout

**Scope:** task_only_diagnostic; scientific; revision `a6bf9a8d59af365fca1e7fa319c512dac16ccca1c2fe71e3e5ef942cbfb56df7`.

At fixed 6,400 updates, full-budget schedules improve sustained ring density compared with the saved 400-horizon baselines.

**Observed:** {'PASS': 3}; wall seconds 178.244; mechanism `bundled_schedule_horizon_intervention`.

ring16_full_schedule6400_v1__r1r2: modes=16, hq=0.97754, mass_tv=0.050293; ring16_full_schedule6400_v1__ka2: modes=16, hq=0.97485, mass_tv=0.054688; ring16_full_schedule6400_v1__k3p: modes=16, hq=0.96338, mass_tv=0.040771

All three full-schedule runs pass the original full ring gate at 6,400 updates: K3P has 66 terminal passing observations, KA2 47 and R1/R2 exactly 5 (required minimum 5). Matched saved 400-horizon baselines have terminal streaks 0/0/3 respectively and all fail. Final covariance errors change K3P 1.599982 to 0.567272, KA2 1.783269 to 0.563027 and R1/R2 0.525637 to 0.423491 (limit 0.85). All endpoints recover 16 modes with mass TV below 0.15 and HQ above 0.85. This supports schedule exhaustion as a major contributor to the ring failures; the coupled intervention does not identify LR alone as the cause. R1/R2 is fragile: update 6,317 fails minimum eigenvalue spread, and its endpoint ratio 0.157181 barely clears 0.15. KA2 last fails minimum spread at 5,617; K3P last fails covariance at 5,300. Every new arm still fails at the sampled 1,600/3,200/4,800 budget cuts. Frozen archived source/runtime, initial tensors/prior, actual target batches, cadence, clean-live sampling and the serialization boundary at 1,600 match. The first 1,600 baseline batches are verified by frozen-law replay and endpoint RNG state; the subsequent 4,800 by the actual baseline trace digest. No baseline is retrained. Source-bound receipts, restored states, final optimizer counts and saved-sample gate recomputations pass. This ring-only diagnostic changes neither original qualification results nor the current leaderboard. The separate clock audit remains unresolved.

**Next:** Conclude this bounded three-run round. Prioritize K3P and KA2 for a future declared full-budget schedule policy comparison across the required task suite before selecting any global recipe. Inspect saved R1/R2 per-component spread and covariance near its late failures before proposing a repair. Treat clock-free eligibility as a separate mechanism question: longer schedules retain explicit time dependence. No automatic additional training, seed study, qualification or default adoption.

[Evidence](../../reports/forge/ring16-schedule-20261011/results.json) · [Record](records/ring16-schedule-20261011-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · ring16-serialized-live-v1

**Scope:** cuda_uninterrupted_serialized_autograd_quality; family_context; revision `e90fc925c996dccff3a054913a040cd4b3b76023972294062139fe5815d7db50`.

Does always-serialized fresh live Ring16 reach the target without restart, and does the previously measured serialization-only401 third trajectory acquire it?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `autograd_execution_order_diagnostic`.



Both CUDA arms complete1600 uninterrupted updates. Always-serialized first passes1400 but fails its one independent confirmation (covariance .894789); then fails1417 and passes final11 checks1434..1600, final covariance .385800. Serializing only401 matches original400 and third-trajectory401 contexts exactly; confirms acquisition867 and passes all45 checks867..1600, final covariance .424426. Identical initial tensors, recipe, prior, target batches, gates and cadence verified against prior continuous truncation/noise; every public numerical package file is unchanged. Truncation/noise acquire earlier and retain their original post-acquisition failures. No indefinite retention, ordinary qualification or default adoption follows.

**Next:** Prioritize deterministic continuous truncation for a separately frozen whole-configuration Tier1 comparison; keep weak-gradient noise second. Serialization reaches late quality and preserves execution mode in checkpoints, but failed first-state confirmation prevents this continuous arm claiming smoke. Do not adopt scheduled401 interventions/restarts or spend automatically on Tier2.

[Evidence](../../reports/forge/ring16-followup/serialized-live-results.json) · [Record](records/ring16-serialized-live-v1.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · ring16-serialized-truncation-v1

**Scope:** cuda_uninterrupted_serialized_truncation_quality; family_context; revision `e90fc925c996dccff3a054913a040cd4b3b76023972294062139fe5815d7db50`.

Do continuous serialized autograd and the fixed measured spectral truncation rule together acquire Ring16 and retain full scheduled quality through1600 without restart?

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `spectral_truncation_and_autograd_execution_order`.



One fresh live CUDA run completes1600 updates, confirms acquisition1050 (independent covariance .817187), and passes every scheduled check1050..1600:34 full passes, zero post-acquisition failures. Final covariance .470678, HQ .956055, massTV .078613 and min eigen ratio .387340 pass all original bounds. Matched initialization/recipe/prior/all1600 target batches/cadence/gates and unchanged public numerical package verified against source-bound continuous truncation and serialization; no controls rerun. Truncation acquires earlier684 and settles834; the combination settles1050 with lower final covariance. Compatibility is demonstrated in this fixed cohort; universal superiority, combined restart parity, indefinite retention and ordinary qualification are not established.

**Next:** Retain continuous truncation and the combination for a separately frozen whole-configuration Tier1 comparison. Preserve failed individual-arm confirmations; no scheduled interventions/restarts, empirical speed ranking, default adoption or automatic Tier2 spend.

[Evidence](../../reports/forge/ring16-followup/serialized-truncation-results.json) · [Record](records/ring16-serialized-truncation-v1.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · tier1-batch-size-v1-readout

**Scope:** task_only_diagnostic; scientific; revision `d079f5df4e8792c33459b08452b31b0a1136cadff2e6f3e59eddf7df890a6200`.

A larger batch may stabilize scalar acquisition and continued learning while retaining ring quality under the unchanged selected BCAP recipe.

**Observed:** {'FAIL': 3, 'PASS': 1}; wall seconds 122.683; mechanism `task_owned_batch_size_and_stationary_hold`.

ring16_acquisition__batch128__hold4000: modes=16, hq=0.9707, mass_tv=0.057373

Batch512 does not repair Gaussian: both batches have longest full passing streak1 and no five-check window through4000. Gaussian final KS is .09594 at128 and .12660 at512. Ring128 passes acquisition and all144/144 hold checks, ending with150 consecutive passes. Ring512 acquires earlier (first five-check window784 versus1584) but has one covariance failure at2384, so its declared strict hold fails143/144 despite passing acquisition and endpoint. All three new CUDA trials complete10400 updates for122.683 new loop seconds, with exact prefix reuse, four exact final restores and676 metric recomputations. No current task, original qualification or calibration result changes.

**Next:** Stop this bounded batch round and retain adopted ring batch128. Gaussian needs a separately declared mechanism comparison; the user favors extrapolation from the past after compaction. Hold batch fixed and use one whole trainer across tasks; judge live acquisition, continued quality and separately declared target-shift response. No automatic larger batch, retry, seed study or new training.

[Evidence](../../reports/forge/tier1-batch-size/results.json) · [Record](records/tier1-batch-size-v1-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · tier1-prior-duration-v1-readout

**Scope:** task_only_diagnostic; scientific; revision `d079f5df4e8792c33459b08452b31b0a1136cadff2e6f3e59eddf7df890a6200`.

The narrow-ring tail covariance may still be converging, and the wider-prior scalar law may become sustained with additional unchanged updates.

**Observed:** {'FAIL': 2, 'PASS': 1}; wall seconds 60.077; mechanism `execution_budget_only`.

ring16_acquisition__mog025-n256__duration1600: modes=16, hq=0.98267, mass_tv=0.046875; ring16_acquisition__mog100-n256__duration1600: modes=16, hq=0.93774, mass_tv=0.058594

The user-requested duration diagnostic completes all three CUDA extensions without repeating prefix updates. Ring16 with 256 particles and MoG sigma .1 passes the full 1600-update gate with terminal suffix 6, covariance error .51431, minimum eigen ratio .38370 and HQ .93774. Ring sigma .025 passes the saved 800/1200-update cuts (suffix 8/16), then loses local eigenvalue spread and finishes FAIL with suffix 1 despite a passing endpoint; first five-pass window at update 750, longest streak 19. Gaussian sigma .1 remains FAIL at 4000 (KS .09594, std ratio 1.74796, mean error .27536 sigma); no five-pass window. Only the execution cap changed; models, optimizers and RNG streams restored exactly. Original 400/1000-update failures and the ordinary 4/6 selection remain intact.

**Next:** Stop the bounded duration round. Ring acquisition supports a separately declared 800-update narrow-MoG question or 1600-update sigma-.1 question with unchanged full bounds; the narrow prior needs an endurance check. Gaussian needs a separately bounded trainer-stability hypothesis rather than more unchanged updates. No automatic further training, calibration claim or default adoption.

[Evidence](../../reports/forge/tier1-prior-duration/results.json) · [Record](records/tier1-prior-duration-v1-readout.json)

### bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9 · tier1-prior-smoke-v1-readout

**Scope:** task_only_diagnostic; scientific; revision `d079f5df4e8792c33459b08452b31b0a1136cadff2e6f3e59eddf7df890a6200`.

Prior cardinality, public cloud/MoG code path and fixed MoG width may determine acquisition and local shape under the selected BCAP trainer. A balanced nine-configuration prior grid keeps all other task and trainer factors fixed.

**Observed:** {'FAIL': 18}; wall seconds 134.613; mechanism `task_owned_prior_hyperparameters`.

ring16_acquisition__cloud-n256: modes=16, hq=0.93042, mass_tv=0.059814

All 18 CUDA cells complete: 0/18 full-quality PASS, 3/18 explicitly narrower smoke-projection PASS. Larger tables regress fixed-budget acquisition. With 256 rows, MoG sigma .1 passes Gaussian location/width at five terminal checks but not sustained KS; ring passes coverage/precision/balance with cloud or MoG .025 but fails full local covariance. No single tested prior passes both original tasks or both proposed smoke questions. Baseline endpoints exactly reproduce the archived BCAP winner. No trainer delta, seed variation, retries or changed historical gates.

**Next:** Stop this finite grid. Keep 256 particles for these budgets; propose task-owned MoG sigma .1 for a separately scoped scalar smoke and .025 for ring acquisition. Preserve original full CDF/covariance questions as quality evidence and calibrate an explicit future profile revision before ordinary qualification. No automatic continuation or defaults.

[Evidence](../../reports/forge/tier1-prior-smoke/results.json) · [Record](records/tier1-prior-smoke-v1-readout.json)

### ka2-slow-prior-0p1 · word-diagnostic-1daeb48c4e66b7ab2e4e1551

**Scope:** task_only_diagnostic; scientific; revision `a3ea26c51800ede1c4dd8983a58b0c2a891bf18e8ef814c7aa5f9283186acf81`.

Conditional parameter-only follow-up after the complete first diagnostic round: hold clean full-horizon optimizer settings fixed and slow the existing learned prior role to prevent code contraction while G/E learns the bijection. Same finite-word joint/inverse core and numerical gates; task-only diagnostic grants no ordinary qualification.

**Observed:** {'PASS': 1}; wall seconds 180.236; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 21/24 observations pass and the passing terminal suffix is 21 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/ka2-slow-prior-0p1.json) · [Record](records/word-diagnostic-1daeb48c4e66b7ab2e4e1551.json)

### k3p-slow-prior-0p25 · word-diagnostic-2079073e755c77a3ee599732

**Scope:** task_only_diagnostic; scientific; revision `917dd6a24d1787ac88dadade867beb7beef5a084150c57285f63b37f4a15b349`.

Conditional parameter-only follow-up after the complete first diagnostic round: hold clean full-horizon optimizer settings fixed and slow the existing learned prior role to prevent code contraction while G/E learns the bijection. Same finite-word joint/inverse core and numerical gates; task-only diagnostic grants no ordinary qualification.

**Observed:** {'FAIL': 1}; wall seconds 152.128; mechanism `floor_constant`.

five_word_joint_acquisition: modes=4, mass_tv=0.22422

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-slow-prior-0p25.json) · [Record](records/word-diagnostic-2079073e755c77a3ee599732.json)

### k3p-coeff1-cap0p1 · word-diagnostic-2cc5a1b19929b60cf60e4434

**Scope:** task_only_diagnostic; scientific; revision `8effaa09e476ee03768abb1abf9282bc3cd18a8969ac2a6444374d449df978b7`.

Independent bounded existing-parameter contrast after low-rate/noise/horizon and slower-prior diagnostics: tune positive K3P critic coefficient/cap strengths on the unchanged 170-dimensional word/latent joint host. Compare against the already-completed clean/full low-rate baseline without rerunning it. No penalty swap, new objective, seed study or qualification bypass credit.

**Observed:** {'FAIL': 1}; wall seconds 150.496; mechanism `floor_constant`.

five_word_joint_acquisition: modes=2, mass_tv=0.62422

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-coeff1-cap0p1.json) · [Record](records/word-diagnostic-2cc5a1b19929b60cf60e4434.json)

### k3p-low-rate-original-noise · word-diagnostic-31c2ec140a756316daa7f398

**Scope:** task_only_diagnostic; scientific; revision `cf5121a07e917a991379d206e74d2c5135ac1dac246df295daa6f644d0d731c5`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'FAIL': 1}; wall seconds 166.127; mechanism `floor_constant`.

five_word_joint_acquisition: modes=4, mass_tv=0.22422

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-low-rate-original-noise.json) · [Record](records/word-diagnostic-31c2ec140a756316daa7f398.json)

### k3p-coeff170-cap1 · word-diagnostic-34636303d73a2e8ff8317fe6

**Scope:** task_only_diagnostic; scientific; revision `7c9ea7c422fdacc4f9bbee35883893cf0d6abb2693341b515598e52b8a47cfa5`.

Independent bounded existing-parameter contrast after low-rate/noise/horizon and slower-prior diagnostics: tune positive K3P critic coefficient/cap strengths on the unchanged 170-dimensional word/latent joint host. Compare against the already-completed clean/full low-rate baseline without rerunning it. No penalty swap, new objective, seed study or qualification bypass credit.

**Observed:** {'PASS': 1}; wall seconds 151.482; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 24/24 observations pass and the passing terminal suffix is 24 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-coeff170-cap1.json) · [Record](records/word-diagnostic-34636303d73a2e8ff8317fe6.json)

### k3p-slow-prior-0p5 · word-diagnostic-4233c39e171cc2256df48c03

**Scope:** task_only_diagnostic; scientific; revision `6ccd6854e1b323c67116a819ebeb65d3edd8331de5510e3df86934e4bc4aa497`.

Conditional parameter-only follow-up after the complete first diagnostic round: hold clean full-horizon optimizer settings fixed and slow the existing learned prior role to prevent code contraction while G/E learns the bijection. Same finite-word joint/inverse core and numerical gates; task-only diagnostic grants no ordinary qualification.

**Observed:** {'FAIL': 1}; wall seconds 150.47; mechanism `floor_constant`.

five_word_joint_acquisition: modes=2, mass_tv=0.6

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-slow-prior-0p5.json) · [Record](records/word-diagnostic-4233c39e171cc2256df48c03.json)

### ka2-slow-prior-0p25 · word-diagnostic-48d0e04cb39975dd920addef

**Scope:** task_only_diagnostic; scientific; revision `1da46ac2e086644934253b01edea57f39b7a099f4a831686647f70c6acd715ec`.

Conditional parameter-only follow-up after the complete first diagnostic round: hold clean full-horizon optimizer settings fixed and slow the existing learned prior role to prevent code contraction while G/E learns the bijection. Same finite-word joint/inverse core and numerical gates; task-only diagnostic grants no ordinary qualification.

**Observed:** {'PASS': 1}; wall seconds 173.749; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 11/24 observations pass and the passing terminal suffix is 11 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/ka2-slow-prior-0p25.json) · [Record](records/word-diagnostic-48d0e04cb39975dd920addef.json)

### r1r2-low-rate-original-roles · word-diagnostic-4d632e29ec48eec7a7cceb41

**Scope:** task_only_diagnostic; scientific; revision `9679c1a73952b37628813de6135e67c35c7f3a482aa05d3bfb8452b4e265d918`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'PASS': 1}; wall seconds 124.176; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 22/24 observations pass and the passing terminal suffix is 9 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/r1r2-low-rate-original-roles.json) · [Record](records/word-diagnostic-4d632e29ec48eec7a7cceb41.json)

### ka2-low-rate-original-noise · word-diagnostic-5b413df2820d3403cf7d166e

**Scope:** task_only_diagnostic; scientific; revision `773d4c9513a9836da5620a64e2c60fbfaebbcce53e02fd761f29bbaee54fc9df`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'FAIL': 1}; wall seconds 179.054; mechanism `floor_constant`.

five_word_joint_acquisition: modes=4, mass_tv=0.22422

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/ka2-low-rate-original-noise.json) · [Record](records/word-diagnostic-5b413df2820d3403cf7d166e.json)

### k3p-slow-prior-0p1 · word-diagnostic-5ef70f1d3085171e255d5880

**Scope:** task_only_diagnostic; scientific; revision `de38e3b1f45be551c22e7cc976398787e4000be7727ec3367abfb83ece929037`.

Conditional parameter-only follow-up after the complete first diagnostic round: hold clean full-horizon optimizer settings fixed and slow the existing learned prior role to prevent code contraction while G/E learns the bijection. Same finite-word joint/inverse core and numerical gates; task-only diagnostic grants no ordinary qualification.

**Observed:** {'FAIL': 1}; wall seconds 151.953; mechanism `floor_constant`.

five_word_joint_acquisition: modes=3, mass_tv=0.4

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-slow-prior-0p1.json) · [Record](records/word-diagnostic-5ef70f1d3085171e255d5880.json)

### ka2-slow-prior-0p5 · word-diagnostic-741f54ec9ea6689d074218a1

**Scope:** task_only_diagnostic; scientific; revision `c90aba418fbbeb49289dedf22dda4dcafdcf493bd38ec2aa2a7bd144a054587a`.

Conditional parameter-only follow-up after the complete first diagnostic round: hold clean full-horizon optimizer settings fixed and slow the existing learned prior role to prevent code contraction while G/E learns the bijection. Same finite-word joint/inverse core and numerical gates; task-only diagnostic grants no ordinary qualification.

**Observed:** {'FAIL': 1}; wall seconds 177.686; mechanism `floor_constant`.

five_word_joint_acquisition: modes=4, mass_tv=0.22422

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/ka2-slow-prior-0p5.json) · [Record](records/word-diagnostic-741f54ec9ea6689d074218a1.json)

### k3p-clean-full-original-rate · word-diagnostic-8698cb431ec5354fb9d66312

**Scope:** task_only_diagnostic; scientific; revision `52107a196a00287e89280e8c39da51816c4a51807eae99d9307441b68c680510`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'FAIL': 1}; wall seconds 177.828; mechanism `floor_constant`.

five_word_joint_acquisition: modes=0, mass_tv=1

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-clean-full-original-rate.json) · [Record](records/word-diagnostic-8698cb431ec5354fb9d66312.json)

### r1r2-mid-rate-fast-roles · word-diagnostic-a89f00328a6b33f44ee3cbb1

**Scope:** task_only_diagnostic; scientific; revision `1aae3017cc71ea978649920a4edbf05da43a0aa0007e82d1a874da75465f207f`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'PASS': 1}; wall seconds 123.07; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 23/24 observations pass and the passing terminal suffix is 22 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/r1r2-mid-rate-fast-roles.json) · [Record](records/word-diagnostic-a89f00328a6b33f44ee3cbb1.json)

### k3p-coeff170-cap0p1 · word-diagnostic-ba0dd91779d34466b64ca9d4

**Scope:** task_only_diagnostic; scientific; revision `a686d1e2e96c93e19cabe769d41584a56e7442a4f30bf27b0c8174c3bfed81e5`.

Independent bounded existing-parameter contrast after low-rate/noise/horizon and slower-prior diagnostics: tune positive K3P critic coefficient/cap strengths on the unchanged 170-dimensional word/latent joint host. Compare against the already-completed clean/full low-rate baseline without rerunning it. No penalty swap, new objective, seed study or qualification bypass credit.

**Observed:** {'PASS': 1}; wall seconds 150.229; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 24/24 observations pass and the passing terminal suffix is 24 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-coeff170-cap0p1.json) · [Record](records/word-diagnostic-ba0dd91779d34466b64ca9d4.json)

### r1r2-low-rate-fast-roles · word-diagnostic-ca65bfdc6b5ccd8b0d43336b

**Scope:** task_only_diagnostic; scientific; revision `d07008027ac5f9882ce257c84386e011e193c424dceb45dcbf63e065286b83ce`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'PASS': 1}; wall seconds 123.315; mechanism `floor_constant`.

five_word_joint_acquisition: modes=5, mass_tv=0.024219

PASS on the unchanged finite five-word joint/inverse task; 22/24 observations pass and the passing terminal suffix is 6 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/r1r2-low-rate-fast-roles.json) · [Record](records/word-diagnostic-ca65bfdc6b5ccd8b0d43336b.json)

### k3p-clean-full-low-rate · word-diagnostic-cbcc08b833cdd7f039b53447

**Scope:** task_only_diagnostic; scientific; revision `f8b63ff6768c3fb3125ec0acb6b03c5f2acb6b37e91a81c433a461e160389ece`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'FAIL': 1}; wall seconds 149.234; mechanism `floor_constant`.

five_word_joint_acquisition: modes=2, mass_tv=0.6

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/k3p-clean-full-low-rate.json) · [Record](records/word-diagnostic-cbcc08b833cdd7f039b53447.json)

### ka2-clean-full-low-rate · word-diagnostic-d15ef7dda68917521973e3c3

**Scope:** task_only_diagnostic; scientific; revision `711b9d749d6e3afedaec2dc28d13901d21912b159e32430ebf59e93dc6f8b553`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'FAIL': 1}; wall seconds 181.77; mechanism `floor_constant`.

five_word_joint_acquisition: modes=4, mass_tv=0.22422

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/ka2-clean-full-low-rate.json) · [Record](records/word-diagnostic-d15ef7dda68917521973e3c3.json)

### ka2-clean-full-original-rate · word-diagnostic-d55cded9da12e599548f1e13

**Scope:** task_only_diagnostic; scientific; revision `6fe8447f6efde0115d44c1e26623fc3fb78eee1ce1e1c778ddfaa78639e40589`.

User-authorized parameter-only diagnosis and repair of existing K3P, KA2 and R1/R2 on the unchanged finite-word joint generator/inverse question. Standalone diagnostic reaches the word task directly; no main-view qualification or default adoption.

**Observed:** {'FAIL': 1}; wall seconds 176.201; mechanism `floor_constant`.

five_word_joint_acquisition: modes=1, mass_tv=0.8

FAIL on the unchanged finite five-word joint/inverse task; 0/24 observations pass and the passing terminal suffix is 0 (required >=5). This reaches the task directly and grants no ordinary Tier 1 or default-adoption credit.

**Next:** Use the completed bounded readout before further research. Preserve original failures and unknown main-view cells; no seed repeat, automatic continuation or default adoption.

[Evidence](../../reports/forge/word-root-cause/receipts/ka2-clean-full-original-rate.json) · [Record](records/word-diagnostic-d55cded9da12e599548f1e13.json)

### word-joint-root-cause-readout · word-diagnostic-readout-46dd1b58096a2b9d65e90c5a

**Scope:** task_only_diagnostic; task_diagnostic_readout; revision `46dd1b58096a2b9d65e90c5ab544a3fb94d66d771f75906b94a8ca80fbff951b`.

Audit the finite five-word joint host and inherited training law before attributing failure to BiGAN or introducing a new stabilization technique. Clean/full K3P/KA2 ablations cross original-parent noise/warmup/horizon activation boundaries; they are task-only diagnostics, not ordinary same-signature configuration searches. Round2/3 preserve their measured clean/full comparator signatures.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `explicit_existing_control_ablation`.



All three existing families pass the unchanged 20,001-update, 24-check/five-terminal-pass word task: selected K3P suffix24, KA2 suffix21, R1/R2 suffix22. Joint pairs/signs/routing are correct. Early network decay with a 100:1 nominal prior/network rate, post-softmax fake-only Gaussian noise and 170-dimensional critic penalty units are explicit host-binding risks; bundled comparisons do not isolate noise as the cause. Lowering prior rate restored KA2; strengthening the existing positive critic coefficient restored K3P. No new loss or spread: all 18 diagnostic arms prior_reg0. A2/direct particle gain were inactive. Historical R1/R2 word was unmeasured, not FAIL. Equality grading and prior_betas consumption bugs were repaired without relabeling old whole verdicts. Coeff170 matches only initial pure-A real R1 scale; fake cap and later kernels remain different. The two coeff170 cap arms have identical saved observations/endpoints, not independent replications or evidence of cap improvement. All arm verdicts/costs reside in their own records; this narrative adds no duplicated task result or cost.

**Next:** Use reports/forge/word-root-cause/README.md and exact configs/forge/selections/word-joint-task-v1.json for scoped reproduction. Ordinary complete-view qualification and calibration remain required before defaults; preserve original failures/unknowns and do not automatically launch another paid study.

[Evidence](../../reports/forge/word-root-cause/summary.json) · [Record](records/word-diagnostic-readout-46dd1b58096a2b9d65e90c5a.json)

### technique-inventory-word-split-v1 · word-split-inventory-v1-readout

**Scope:** source_bound_campaign_readout; scientific; revision `cbb19c5e55e93aff93abd4092bbbe79e10c03a8b9f65a3f5db3b97f0f557d1ec`.

A confirmed can-it-pass joint five-word smoke separates acquisition from continuing hold while preserving all generation and inverse bounds.

**Observed:** No normalized scientific verdict; wall seconds unknown; mechanism `task_gate_reform`.



Five runnable selected families each pass 5/6 required Tier 1 tasks and all acquire five words. BCAP fails only Gaussian; K3P, KA2, R1/R2 and released GAN v3 MoG fail only ring16. E22/Atlas remain preflight-blocked. No ordinary Tier 2 work is eligible. All 35 CUDA attempts complete for 1796.544469 paid seconds; all 31 available scalar endpoint comparisons match the previous post-truncation cohort. No scientific retries or source pooling.

**Next:** Keep the acquisition/hold split. The prepared constant positive-smoothing BCAP comparison is the next bounded option; preserve full smoke gates and require all six passes before Tier 2. No search or default adoption follows automatically.

[Evidence](../../reports/forge/word-split-inventory/readout.json) · [Record](records/word-split-inventory-v1-readout.json)

## Compact publications and closed partial cuts

These study and trial projections preserve recorded outcomes, unknowns and source cohorts. They do not regrade archived science or replace original receipts. Overlapping study/trial costs must not be summed.

[Normalized publication records](publication-records.json). Search exact IDs, failed requirement metric names, mechanisms and goals with `forge recall`. Check current input coverage with `forge compile --check`.

- **bcap-tier2-search-v1--7**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--7.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-global-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-global-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-particle-rownorm-only-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-particle-rownorm-only-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **word-split-inventory-ka2-v1**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/word-split-inventory-ka2-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **pr223-original-full19-retest**: Closed original19 cut: 5/19 PASS, 0 FAIL, 14 unavailable. Final accepted status INCOMPLETE; required execution counts {'NOT_RUN': 14, 'PASS': 5}. Historical positives and overlapping cut costs are not pooled. [Source](../../reports/forge/pr223-original-full-retest-prefix5-20261004/results.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier1-refresh-v1**: Concluded 8 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier1-refresh-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **r1r2-modern-family-round1-v1**: Concluded 8 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/r1r2-modern-family-round1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **policy-family-defaults-round2-v1**: Concluded 8 configurations; recorded outcome incomplete_comparison. Original terminal gates and study persistence gates remain distinct. [Source](../../reports/forge/policy-family-inventory.json) · [Board](../../reports/forge/policy-family-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-global-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-global-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **policy-family-defaults-round3-v1**: Concluded 8 configurations; recorded outcome incomplete_comparison. Original terminal gates and study persistence gates remain distinct. [Source](../../reports/forge/policy-family-inventory.json) · [Board](../../reports/forge/policy-family-inventory.md)
- **pr223-original-full19-retest**: Closed original19 cut: 16/19 PASS, 0 FAIL, 3 unavailable. Final accepted status INCOMPLETE; required execution counts {'INVALID': 1, 'NOT_RUN': 2, 'PASS': 16}. Historical positives and overlapping cut costs are not pooled. [Source](../../reports/forge/pr223-original-full-retest-stopped17-20261004/results.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-prior-mmd-search-v1**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-prior-mmd-search-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-six-smoothing-v1**: Concluded 3 whole configurations. Recorded selection: qualified_winner; default adoption False. [Source](../../reports/forge/configuration-search/bcap-six-smoothing-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-particle-rownorm-only-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-particle-rownorm-only-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-sgda-tier1-v1**: Concluded 5 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-sgda-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-k3p-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-k3p-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-ka2-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-ka2-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **ka2-tier1-refresh-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/ka2-tier1-refresh-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-sgda-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-sgda-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **word-split-inventory-bcap-dualnorm-v1**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/word-split-inventory-bcap-dualnorm-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **r1r2-tier1-refresh-v1**: Concluded 12 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/r1r2-tier1-refresh-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-global-input-noise-tier1-v1**: Concluded 8 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-global-input-noise-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **policy-family-defaults-round1-cli-recovery-v2**: Concluded 8 configurations; recorded outcome incomplete_comparison. Original terminal gates and study persistence gates remain distinct. [Source](../../reports/forge/policy-family-inventory.json) · [Board](../../reports/forge/policy-family-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-d-only-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-d-only-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-ada-nsgda-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-ada-nsgda-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-particle-rownorm-only-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-particle-rownorm-only-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-k3p-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-k3p-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-ada-nsgda-tier1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-ada-nsgda-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **tier1-completion-v1-k3p**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/tier1-completion-v1-k3p.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-r1r2-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-r1r2-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-layer-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-layer-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-tier1-refresh-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-tier1-refresh-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-global-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-global-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-ada-nsgda-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-ada-nsgda-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-layer-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-layer-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-dualnorm-pacing-v2-a**: Concluded 20 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-dualnorm-pacing-v2-a.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-dualnorm-momentum-tier1-v1**: Concluded 8 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-dualnorm-momentum-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--11**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--11.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-family-defaults-round1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-family-defaults-round1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-ada-nsgda-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-ada-nsgda-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-nsgda-layer-tier1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-nsgda-layer-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--13**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--13.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-sgda-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-sgda-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **pure-bcap-non-saturating-joint-rates-v2**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/pure-bcap-non-saturating-joint-rates-v2.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-release07-gan-v3-mog-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-release07-gan-v3-mog-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-ada-nsgda-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-ada-nsgda-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-global-direct-moments-tier1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-global-direct-moments-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--5**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--5.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--1.json) · [Board](../../reports/forge/technique-inventory.md)
- **r1r2-global-repair-rates-v1**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/r1r2-global-repair-rates-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-k3p-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-k3p-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--0**: Concluded 24 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--0.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-global-repair-rates-v1**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-global-repair-rates-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--6**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--6.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-global-tier1-v3**: Concluded 8 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-global-tier1-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **tier1-completion-v1-ka2**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/tier1-completion-v1-ka2.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-r1r2-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-r1r2-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--12**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--12.json) · [Board](../../reports/forge/technique-inventory.md)
- **word-split-inventory-r1r2-v1**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/word-split-inventory-r1r2-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-prior-sigreg-search-v1**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-prior-sigreg-search-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **pure-bcap-relativistic-rates-v1**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/pure-bcap-relativistic-rates-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-layer-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-layer-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **r1r2-modern-toy-v1**: Concluded 4 whole configurations. Recorded selection: qualified_winner; default adoption False. [Source](../../reports/forge/configuration-search/r1r2-modern-toy-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-dualnorm-d-only-tier1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-dualnorm-d-only-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **family-winner-round1**: 60 configurations; finite_screen_complete_no_fully_qualified_config. Capacity, a smoke pass or a good final endpoint does not qualify a whole configuration. Calibration and separately registered confirmation/robustness remain required. External GPU contention disables speed ranking. [Source](../../reports/forge/family-winner-round1/campaign-completion.json) · [Board](../../reports/forge/family-winner-round1/README.md)
- **pure-bcap-least-squares-joint-rates-v2**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/pure-bcap-least-squares-joint-rates-v2.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--8**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--8.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-sgda-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-sgda-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **k3p-global-tier1-v2**: Concluded 12 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/k3p-global-tier1-v2.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--10**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--10.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-particle-rownorm-only-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-particle-rownorm-only-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-ka2-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-ka2-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--9**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--9.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-release07-gan-v3-mog-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-release07-gan-v3-mog-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-nsgda-global-tier1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-nsgda-global-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-overnight-search-v1--0**: Concluded 12 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-overnight-search-v1--0.json) · [Board](../../reports/forge/technique-inventory.md)
- **api-model-toys-v1**: Three PASS (both AE cases and categorical VAE), three full-budget FAIL (hard VAE terminal stability and both Gaussian chains), two BLOCKED bimodal full runs. AE-DDGAN paired reconstruction passes its terminal subgates while independent generation fails. Three-update deeper GIFs are INCOMPLETE software prefixes. No qualification or default credit. [Source](../../reports/toy_audit/api_contract/model_modes/publication.json) · [Board](../../reports/forge/technique-inventory.md)
- **pr223-native3-continuation**: Native3 continuation: 0/3 PASS, 0 FAIL, 3 unavailable; final accepted INCOMPLETE. Original19 parent remains 16 PASS/1 INVALID/2 NOT_RUN under its own source. Current cost includes prior case debit once and cumulative metadata once. [Source](../../reports/forge/pr223-native3-first-invalid-20261004/results.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-release07-gan-v3-mog-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-release07-gan-v3-mog-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **pure-bcap-hinge-joint-rates-v2**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/pure-bcap-hinge-joint-rates-v2.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-particle-rownorm-only-tier1-v1**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-particle-rownorm-only-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **release07-gan-v3-mog-tier1-refresh-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/release07-gan-v3-mog-tier1-refresh-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-global-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-global-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-d-only-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-d-only-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-d-only-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-d-only-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-d-only-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-d-only-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **pr223-native3-repaired-continuation**: Repaired native3 continuation: 3/3 PASS, 0 FAIL, 0 unavailable; final accepted PASS. Original19 parent remains 16 PASS/1 INVALID/2 NOT_RUN under its own source. Current cost includes original19 and pretraining-invalid debits separately once and SAME cumulative metadata once. [Source](../../reports/forge/pr223-native3-repaired-20261004/results.json) · [Board](../../reports/forge/technique-inventory.md)
- **tier1-completion-v1-bcap**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/tier1-completion-v1-bcap.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--15**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--15.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--3**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--3.json) · [Board](../../reports/forge/technique-inventory.md)
- **word-split-inventory-release07-gan-v3-mog-v1**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/word-split-inventory-release07-gan-v3-mog-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **pure-bcap-wasserstein-joint-rates-v2**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/pure-bcap-wasserstein-joint-rates-v2.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-dualnorm-zero-tier1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-dualnorm-zero-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **tier1-completion-v1-r1r2**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/tier1-completion-v1-r1r2.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-family-defaults-round1-v1**: Concluded 8 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-family-defaults-round1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **release07-gan-v3-mog-family-defaults-round1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/release07-gan-v3-mog-family-defaults-round1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-optim-adam-tier1-v1**: Concluded 5 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-optim-adam-tier1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-dualnorm-pacing-v2-c-4054018e4d6380ed**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-dualnorm-pacing-v2-c-4054018e4d6380ed.json) · [Board](../../reports/forge/technique-inventory.md)
- **ka2-family-defaults-round1-v1**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/ka2-family-defaults-round1-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-nsgda-layer-v6**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-nsgda-layer-v6.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-ka2-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-ka2-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-overnight-search-v1--1**: Concluded 12 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-overnight-search-v1--1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-prior-covariance-search-v1**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-prior-covariance-search-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **word-split-inventory-k3p-v1**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/word-split-inventory-k3p-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-r1r2-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-r1r2-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-prior-coulomb-search-v1**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-prior-coulomb-search-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-prior-proximal-search-v1**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-prior-proximal-search-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **tier1-completion-v1-release07-gan-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/tier1-completion-v1-release07-gan-v3.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-dualnorm-v5**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-dualnorm-v5.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--4**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--4.json) · [Board](../../reports/forge/technique-inventory.md)
- **policy-family-defaults-round4-prior-balance-v1**: Concluded 8 configurations; recorded outcome incomplete_comparison. Original terminal gates and study persistence gates remain distinct. [Source](../../reports/forge/policy-family-inventory.json) · [Board](../../reports/forge/policy-family-inventory.md)
- **ka2-global-repair-rates-v1**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/ka2-global-repair-rates-v1.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--14**: Concluded 2 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--14.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-sgda-v4**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-sgda-v4.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-dualnorm-pacing-v2-b-006d03f6c1648102**: Concluded 3 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-dualnorm-pacing-v2-b-006d03f6c1648102.json) · [Board](../../reports/forge/technique-inventory.md)
- **bcap-tier2-search-v1--2**: Concluded 4 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/bcap-tier2-search-v1--2.json) · [Board](../../reports/forge/technique-inventory.md)
- **gaussian-smoke-inventory-bcap-v3**: Concluded 1 whole configurations. Recorded selection: best_observed; default adoption False. [Source](../../reports/forge/configuration-search/gaussian-smoke-inventory-bcap-v3.json) · [Board](../../reports/forge/technique-inventory.md)

## Unresolved imports and limitations

- **structured_mapping_scope:** These files are inventoried support/config/evidence; no scientific result is inferred from unrecognized schemas. (2287 classified sources.)
- **structured_mapping_scope:** These files are inventoried support/config/evidence; no scientific result is inferred from unrecognized schemas. (130 classified sources.)
- **structured_mapping_scope:** These files are inventoried support/config/evidence; no scientific result is inferred from unrecognized schemas. (7 classified sources.)
- **dt075-14k:** Review reports grid/staggered 14k failures after stationarity-ladder release; exact package/fixture/results not located in pinned #155.
- **ema995-dtracking:** Review reports EMA .995 D-tracking 3/3; exact later live/EMA paired receipts are unbound.
- **center-sensitivity:** Review reports sub-0.03 sigma centre-error sensitivity; original scope and receipts remain unbound.
- **local-artifacts:** Historical /ml2 checkpoints, samples and runtimes have not been imported or assumed available.

## Compilation provenance

Reducer `forge-knowledge-v2`; input digest `fd56e2bab9d2f16539850aa615700d1cbfa5d923d5dec3dc1d32ad240e5cb39d`. [Full input hashes and coverage](compilation.json). No training or image inspection occurs during compilation.
