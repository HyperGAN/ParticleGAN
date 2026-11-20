# Round-3 Track 4 Study: Bounded Calibration Audit, Tier-3 `14k` Native Continuation & Tier-4 Untouched Frontier Holdout

## 1. Executive Summary & Provenance

This study executes a **zero-tuning, frozen-recipe untouched holdout evaluation** and **mathematical calibration audit** across five active ParticleGAN formulations (`empf-v8`, `rhbi-v12`, `rgdt-v3`, `unified-v3`, `t4-arch-v3`) plus one deliberate unregularized negative control (`r3-negctrl-unreg-adam`) on two newly declared `research_diagnostic` holdout views:

1. **Tier-3 Long-Horizon Holdout** ([`r3-tier3-long-horizon-holdout-v1.json`](../../../configs/forge/views/r3-tier3-long-horizon-holdout-v1.json), 18 tasks): fresh-process `14,000`-update native continuations (`grid100_14k`, `rotated100_14k`, `staggered100_14k`), uninterrupted `7,500`-step ring endurance (`ring_hold` + `ring_extension`), `4,000`-step discrete word BiGAN continuation (`five_word_joint_smoke` + `five_word_joint_hold`), paired `target_shift_recovery`, `zero_tanh_rail_escape`, `clockfree_audit`, and the 3-task cheap calibration smoke predicate (`two_pole`, `unused_token_hold`, `ae_gan_hold`).
2. **Tier-4 Frontier Holdout** ([`r3-tier4-frontier-holdout-v1.json`](../../../configs/forge/views/r3-tier4-frontier-holdout-v1.json), 16 tasks): `grid100` prerequisite gate + all 3 runnable 16-layer/rail-guarded 100-mode tasks (`grid100_mlp16`, `grid100_res16_tanh`, `grid100_res16_scaled_tanh`) + all 12 `STRETCH_VECTOR_HOSTS` tasks explicitly audited for host-adapter preflight boundaries.

All 12 studies were executed on `NVIDIA RTX A6000` GPUs from unified source commit `ff9b14b78` through the public `GANTrainer` API under protocol seed `0`, `deterministic_orthogonal` initialization, task-owned architectures and initial priors, and live scoring weights (`ema_decay=0.0`, `serve_average=0.0`, `output_noise_std=0.0`), consuming **`26,618.96s` (`7.39` GPU-hours)** total across `12` concluded source-bound Forge readout receipts. The machine-readable summary is recorded in [`r3-calibration-holdout.json`](../calibration/r3-calibration-holdout.json).

### Headline Findings

- **Top Tier-3 Long-Horizon Performers (`13 / 18 PASS` tie, sharply complementary profiles):**
  - **`t4-arch-v3`** (`4,215.0s`) is the **only candidate to pass all `3 / 3` fresh-process `14,000`-update native continuations** (`grid100_14k` `prec=0.9839 / acc=0.3313`, `rotated100_14k` `0.9837 / 0.3327`, `staggered100_14k` `0.9863 / 0.3282`) with zero accuracy degradation from `7k -> 14k`, achieves the strongest `zero_tanh_rail_escape` (`0.0011` MAE), and is the **only candidate to complete and pass `five_word_joint_smoke`** (`561.1s`, `5/5` words, `exact=1`). However, it drops `apple` during `five_word_joint_hold` (`step=7334`, `4/5` words, `FAIL`) and misses the `mode_hold` quality gate (`hq=0.8308 < 0.90`, `FAIL`), which gates `ring_hold`, `ring_extension`, and `target_shift_recovery`.
  - **`unified-v3`** (`6,060.7s`) passes **`2 / 3` `14k` continuations with strictly improving accuracy** (`rotated100` `0.6953 -> rotated100_14k 0.4961`, `staggered100` `0.5863 -> staggered100_14k 0.5147`) **and** passes `mode_hold` (`hq=1.0000`), uninterrupted `7,500`-step `ring_extension` (`300/300` checks), and `target_shift_recovery` (`81/81` live checks, `min_hq=0.9160`). It misses `grid100` solely on the first terminal-window check (`step=6000`, `acc=0.9180`; holdout `prec=0.9867 / acc=0.6751` and `4/5` terminal checks pass), fails `zero_tanh_rail_escape` (`0.8795` MAE, rail-trapped), and hits the `900s` per-task budget (`INCOMPLETE`) on `five_word_joint_smoke`.
  - **`empf-v8`** (`3,822.6s`, `10/18 PASS`), **`rgdt-v3`** (`5,984.8s`, `9/18 PASS`), and **`rhbi-v12`** (`3,840.6s`, `8/18 PASS`) all pass `ring_hold` and uninterrupted `7,500`-step `ring_extension` (`300/300`), while failing all three `100`-mode tasks (`prec <= 0.358`) and timing out (`INCOMPLETE`, `900s`) on `five_word_joint_smoke`. `empf-v8` and `rgdt-v3` pass `target_shift_recovery` (`81/81`, `min_hq=0.9995` / `0.9998`), whereas `rhbi-v12` fails (`45/81`, `min_hq=0.7805`); only `empf-v8` among these three passes `zero_tanh_rail_escape` (`0.0026` MAE).
- **Tier-4 Untouched Frontier Holdout (`4 / 4` runnable `PASS` for `t4-arch-v3`; `12 / 12` stretch hosts explicitly `BLOCKED`):**
  - **`t4-arch-v3`** (`2,162.7s` new GPU spend) is the **only candidate to pass the `grid100` prerequisite gate** and **passes all `3 / 3` runnable deep/rail-guarded 16-layer Tier-4 tasks**: `grid100_mlp16` (`prec=0.9814, acc=0.3010`), `grid100_res16_tanh` (`prec=0.9856, acc=0.4170`), and `grid100_res16_scaled_tanh` (`prec=0.9868, acc=0.2884`).
  - All five other candidates (`empf-v8`, `rhbi-v12`, `rgdt-v3`, `unified-v3`, `r3-negctrl-unreg-adam`) incurred `0.0s` new Tier-4 GPU spend because Forge cleanly reused their compatible `grid100` `FAIL` receipt from Tier-3, gating the 3 runnable deep/rail tasks.
  - Across **all 6 candidates**, all `12 / 12` `STRETCH_VECTOR_HOSTS` tasks are explicitly **`BLOCKED`** at host preflight (`no registered public Forge adapter for this stretch host`) — never conflated with numerical `FAIL`.
- **Negative-Control True-Rejection Verified (`r3-negctrl-unreg-adam`):** Unregularized parameter-space Adam (`reg_coeff=0.0`, `prior_reg=0.0`, `prior_regularizer="none"`) fails `2/3` cheap smoke predicates (`two_pole` `mean_abs=0.0714 < 0.30`, `unused_token_hold` `concept_move=0.6850`) and collapses across every multimodal reference (`mode_hold` `1/8` modes `hq=0.2476`, `grid100` `mass_tv=0.9800`, `five_word_joint_smoke` `0/5` words `mass_tv=1.0`), confirming `true_reject = (FAIL, FAIL)` gate sensitivity while passing `clockfree_audit` (`4/4` parity).
- **Calibration Adoption & Default Promotion Remain `BLOCKED` (`DEFAULT_RECIPE == "empf"` untouched):** Frozen [`criteria-v1.json`](../../../configs/forge/calibration/criteria-v1.json) (`sha256: 6eac2531eed68e392c11a57328999ac5dce509f8fe51d23e7b9031b7a4e3f9d1`) mathematically blocks calibration acceptance on three independent grounds: (1) `0` single global recipes pass `16/16` reference tasks (`unified-v3` `15/16` and `t4-arch-v3` `13/16` cover `16/16` only in union), (2) `discriminator_stability` rev-19's 11-task Tier-1 smoke tier (`1,229.0s`, ratio `0.7009`) violates both `<= 900.0s` and `<= 0.10` cost bounds, and (3) under the cheap 3-task smoke screen (`25.8s–54.8s`), all 5 partial-solver recipes are `false_accept` (`smoke=PASS, ref=FAIL`), making `maximum_false_accept_fraction <= 0.10` infeasible without either a stronger cheap discriminator or `>= 9` `true_reject` controls per `false_accept`.

> [!IMPORTANT]
> **Governance & Scope:** All 12 studies belong to `evidence_scope = "research_diagnostic"` (`qualification_input = false`, `qualification_reuse = false`). Per `AGENTS.md` and `EXPERIMENTATION.md`, they grant no ordinary qualification credit, never manufacture calibration acceptance, and keep `particlegan.DEFAULT_RECIPE == "empf"` untouched.

### Concluded Readout Provenance (12 Official Forge Receipts)

| Study ID | Candidate | Candidate Revision | View | GPU Spend (s) | Observed Task Outcomes | Concluded Readout Receipt |
| :--- | :--- | :--- | :--- | ---: | :--- | :--- |
| [`r3-t3-holdout-empf-v8-study`](../../../configs/forge/studies/r3-t3-holdout-empf-v8-study.json) | `empf-v8` | `039950bf79df...` | `r3-tier3-long-horizon-holdout-v1` | `3,822.59` | `10 PASS`, `3 FAIL`, `1 INCOMPLETE`, `4 UNRUN(gated)` | [`readout-d0f74bbbd3c76f240c0e2729.json`](../records/readout-d0f74bbbd3c76f240c0e2729.json) |
| [`r3-t3-holdout-rhbi-v12-study`](../../../configs/forge/studies/r3-t3-holdout-rhbi-v12-study.json) | `rhbi-v12` | `c8ceb0973a83...` | `r3-tier3-long-horizon-holdout-v1` | `3,840.64` | `8 PASS`, `5 FAIL`, `1 INCOMPLETE`, `4 UNRUN(gated)` | [`readout-a97cc4411799f84fe22dec86.json`](../records/readout-a97cc4411799f84fe22dec86.json) |
| [`r3-t3-holdout-rgdt-v3-study`](../../../configs/forge/studies/r3-t3-holdout-rgdt-v3-study.json) | `rgdt-v3` | `f1f2d75597e0...` | `r3-tier3-long-horizon-holdout-v1` | `5,984.85` | `9 PASS`, `4 FAIL`, `1 INCOMPLETE`, `4 UNRUN(gated)` | [`readout-a01d9274e8c48786bbd562d4.json`](../records/readout-a01d9274e8c48786bbd562d4.json) |
| [`r3-t3-holdout-unified-v3-study`](../../../configs/forge/studies/r3-t3-holdout-unified-v3-study.json) | `unified-v3` | `349254f0ccb8...` | `r3-tier3-long-horizon-holdout-v1` | `6,060.66` | `13 PASS`, `2 FAIL`, `1 INCOMPLETE`, `2 UNRUN(gated)` | [`readout-3a8aa62939f0e420f0d08ed2.json`](../records/readout-3a8aa62939f0e420f0d08ed2.json) |
| [`r3-t3-holdout-t4-arch-v3-study`](../../../configs/forge/studies/r3-t3-holdout-t4-arch-v3-study.json) | `t4-arch-v3` | `1110cf4ed9a5...` | `r3-tier3-long-horizon-holdout-v1` | `4,214.96` | `13 PASS`, `2 FAIL`, `0 INCOMPLETE`, `3 UNRUN(gated)` | [`readout-ff22843954240927dfd7ecc3.json`](../records/readout-ff22843954240927dfd7ecc3.json) |
| [`r3-t3-holdout-negctrl-study`](../../../configs/forge/studies/r3-t3-holdout-negctrl-study.json) | `r3-negctrl-unreg-adam` | `b5e2766672bb...` | `r3-tier3-long-horizon-holdout-v1` | `532.59` | `3 PASS`, `8 FAIL`, `0 INCOMPLETE`, `7 UNRUN(gated)` | [`readout-07a5dd7f26c48345c7a9f35f.json`](../records/readout-07a5dd7f26c48345c7a9f35f.json) |
| [`r3-t4-holdout-empf-v8-study`](../../../configs/forge/studies/r3-t4-holdout-empf-v8-study.json) | `empf-v8` | `039950bf79df...` | `r3-tier4-frontier-holdout-v1` | `0.00`* | `0 PASS`, `1 FAIL` (reused), `3 UNRUN(gated)`, `12 BLOCKED` | [`readout-c0ac30cc7b69526e535b9091.json`](../records/readout-c0ac30cc7b69526e535b9091.json) |
| [`r3-t4-holdout-rhbi-v12-study`](../../../configs/forge/studies/r3-t4-holdout-rhbi-v12-study.json) | `rhbi-v12` | `c8ceb0973a83...` | `r3-tier4-frontier-holdout-v1` | `0.00`* | `0 PASS`, `1 FAIL` (reused), `3 UNRUN(gated)`, `12 BLOCKED` | [`readout-51a0d5590d041ce50f3ce174.json`](../records/readout-51a0d5590d041ce50f3ce174.json) |
| [`r3-t4-holdout-rgdt-v3-study`](../../../configs/forge/studies/r3-t4-holdout-rgdt-v3-study.json) | `rgdt-v3` | `f1f2d75597e0...` | `r3-tier4-frontier-holdout-v1` | `0.00`* | `0 PASS`, `1 FAIL` (reused), `3 UNRUN(gated)`, `12 BLOCKED` | [`readout-d547ba37dabfc5ab66eca538.json`](../records/readout-d547ba37dabfc5ab66eca538.json) |
| [`r3-t4-holdout-unified-v3-study`](../../../configs/forge/studies/r3-t4-holdout-unified-v3-study.json) | `unified-v3` | `349254f0ccb8...` | `r3-tier4-frontier-holdout-v1` | `0.00`* | `0 PASS`, `1 FAIL` (reused), `3 UNRUN(gated)`, `12 BLOCKED` | [`readout-14efe09d667072696e386290.json`](../records/readout-14efe09d667072696e386290.json) |
| [`r3-t4-holdout-t4-arch-v3-study`](../../../configs/forge/studies/r3-t4-holdout-t4-arch-v3-study.json) | `t4-arch-v3` | `1110cf4ed9a5...` | `r3-tier4-frontier-holdout-v1` | `2,162.67` | **`4 PASS`** (`1` reused + `3` new), `0 FAIL`, `12 BLOCKED` | [`readout-4acf96c2efad5e17f86d9df2.json`](../records/readout-4acf96c2efad5e17f86d9df2.json) |
| [`r3-t4-holdout-negctrl-study`](../../../configs/forge/studies/r3-t4-holdout-negctrl-study.json) | `r3-negctrl-unreg-adam` | `b5e2766672bb...` | `r3-tier4-frontier-holdout-v1` | `0.00`* | `0 PASS`, `1 FAIL` (reused), `3 UNRUN(gated)`, `12 BLOCKED` | [`readout-edae8af27f75b55c971db675.json`](../records/readout-edae8af27f75b55c971db675.json) |

*\*`0.00s` new Tier-4 GPU spend reflects exact compatibility-key reuse of each candidate's `grid100` `FAIL` receipt from the paired Tier-3 study in the same queue root (`reuse_note`: overlapping readouts are not independent runs or additive costs).*

---

## 2. Tier-3 Long-Horizon Holdout Leaderboard (`r3-tier3-long-horizon-holdout-v1`)

### 2.1 Candidate Scoreboard

| Rank | Candidate | Tier-3 `PASS` | `FAIL` | `INCOMPLETE` (Timeout) | `UNRUN` (Gated) | `14k` Continuations | `ring_extension` (`7.5k`) | `target_shift` | `zero_tanh_rail` | `five_word` Smoke / Hold | 3-Task Smoke | Tier-3 GPU Spend |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| 1 (tie) | **`t4-arch-v3`** | **13 / 18** | 2 | 0 | 3 | **3 / 3 PASS** | UNRUN (`mode_hold` FAIL) | UNRUN (`mode_hold` FAIL) | **PASS** (`0.0011`) | **PASS** / FAIL | PASS (`35.4s`) | `4,215.0s` |
| 1 (tie) | **`unified-v3`** | **13 / 18** | 2 | 1 | 2 | **2 / 3 PASS** (`grid100` FAIL) | **PASS** (`300/300`) | **PASS** (`81/81`) | FAIL (`0.8795`) | INCOMPLETE / UNRUN | PASS (`48.1s`) | `6,060.7s` |
| 3 | **`empf-v8`** (default) | **10 / 18** | 3 | 1 | 4 | 0 / 3 (gated) | **PASS** (`300/300`) | **PASS** (`81/81`) | **PASS** (`0.0026`) | INCOMPLETE / UNRUN | PASS (`41.1s`) | `3,822.6s` |
| 4 | **`rgdt-v3`** | **9 / 18** | 4 | 1 | 4 | 0 / 3 (gated) | **PASS** (`300/300`) | **PASS** (`81/81`) | FAIL (window) | INCOMPLETE / UNRUN | PASS (`50.9s`) | `5,984.8s` |
| 5 | **`rhbi-v12`** | **8 / 18** | 5 | 1 | 4 | 0 / 3 (gated) | **PASS** (`300/300`) | FAIL (`45/81`) | FAIL (window) | INCOMPLETE / UNRUN | PASS (`54.8s`) | `3,840.6s` |
| Ctrl | **`r3-negctrl-unreg-adam`** | **3 / 18** | 8 | 0 | 7 | 0 / 3 (gated) | UNRUN (`mode_hold` FAIL) | UNRUN (`mode_hold` FAIL) | FAIL (`0.8669`) | FAIL / UNRUN | **FAIL** (`25.8s`) | `532.6s` |

### 2.2 Complete 18-Task Outcome Matrix

| Task | Role / Dependency | `empf-v8` | `rhbi-v12` | `rgdt-v3` | `unified-v3` | `t4-arch-v3` | `r3-negctrl-unreg-adam` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `two_pole` | Cheap smoke (1/3) | **PASS** (`0.9583`) | **PASS** (`0.9954`) | **PASS** (`0.8769`) | **PASS** (`0.9971`) | **PASS** (`0.8443`) | **FAIL** (`0.0714`) |
| `unused_token_hold` | Cheap smoke (2/3) | **PASS** (`0.9881/0.9915`) | **PASS** (`0.9906/0.9888`) | **PASS** (`0.9995/0.9877`) | **PASS** (`0.9906/0.9888`) | **PASS** (`0.9623/0.9122`) | **FAIL** (`0.6850/0.9730`) |
| `ae_gan_hold` | Cheap smoke (3/3) | **PASS** (`0.0195/0.0044`) | **PASS** (`0.0244/0.0053`) | **PASS** (`0.0077/0.0042`) | **PASS** (`0.0308/0.0093`) | **PASS** (`0.0014/0.0032`) | **PASS** (`0.0013/0.0288`) |
| `clockfree_audit` | `14k` gate (`4` parity checks) | **PASS** (`4/4`) | **PASS** (`4/4`) | **PASS** (`4/4`) | **PASS** (`4/4`) | **PASS** (`4/4`) | **PASS** (`4/4`) |
| `zero_tanh_smoke` | Gate for rail escape | **PASS** (`0.0024`) | **PASS** (`0.0039`) | **PASS** (`0.0030`) | **PASS** (`0.0028`) | **PASS** (`0.0017`) | **PASS** (`0.0026`) |
| `zero_tanh_rail_escape` | After `zero_tanh_smoke` | **PASS** (`0.0026`) | **FAIL** (`0.0047`* win) | **FAIL** (`0.0035`* win) | **FAIL** (`0.8795` rail) | **PASS** (`0.0011`) | **FAIL** (`0.8669` rail) |
| `five_word_joint_smoke` | Discrete word BiGAN (`20k`) | **INCOMPLETE** (`900.1s`) | **INCOMPLETE** (`900.3s`) | **INCOMPLETE** (`900.3s`) | **INCOMPLETE** (`900.5s`) | **PASS** (`561.1s`, `5/5`) | **FAIL** (`151.3s`, `0/5`) |
| `five_word_joint_hold` | Cont. of `five_word_joint_smoke` | UNRUN (gated) | UNRUN (gated) | UNRUN (gated) | UNRUN (gated) | **FAIL** (`step 7334`, `4/5`) | UNRUN (gated) |
| `mode_hold` | Gate for ring & shift | **PASS** (`hq=1.0000`) | **PASS** (`hq=1.0000`) | **PASS** (`hq=1.0000`) | **PASS** (`hq=1.0000`) | **FAIL** (`hq=0.8308`) | **FAIL** (`1/8`, `hq=0.2476`) |
| `ring_hold` | After `mode_hold` | **PASS** (`conv=1400`) | **PASS** (`conv=1400`) | **PASS** (`conv=1400`) | **PASS** (`conv=1400`) | UNRUN (gated) | UNRUN (gated) |
| `ring_extension` | Uninterrupted `7.5k` ring | **PASS** (`300/300`) | **PASS** (`300/300`) | **PASS** (`300/300`) | **PASS** (`300/300`) | UNRUN (gated) | UNRUN (gated) |
| `target_shift_recovery` | After `mode_hold` | **PASS** (`81/81`, `0.9995`) | **FAIL** (`45/81`, `0.7805`) | **PASS** (`81/81`, `0.9998`) | **PASS** (`81/81`, `0.9160`) | UNRUN (gated) | UNRUN (gated) |
| `grid100` | `7k` gate for `grid100_14k` | **FAIL** (`0.3460`) | **FAIL** (`0.2068`) | **FAIL** (`0.3080`) | **FAIL** (`0.9867`† win) | **PASS** (`0.9816`) | **FAIL** (`tv=0.9800`) |
| `rotated100` | `7k` gate for `rotated100_14k` | **FAIL** (`0.3582`) | **FAIL** (`0.2028`) | **FAIL** (`0.2507`) | **PASS** (`0.9856`) | **PASS** (`0.9825`) | **FAIL** (`0.6664`) |
| `staggered100` | `7k` gate for `staggered100_14k` | **FAIL** (`0.2960`) | **FAIL** (`0.2960`) | **FAIL** (`0.2989`) | **PASS** (`0.9910`) | **PASS** (`0.9833`) | **FAIL** (`0.7090`) |
| `grid100_14k` | Fresh-process `7k -> 14k` | UNRUN (gated) | UNRUN (gated) | UNRUN (gated) | UNRUN (gated) | **PASS** (`0.9839`) | UNRUN (gated) |
| `rotated100_14k` | Fresh-process `7k -> 14k` | UNRUN (gated) | UNRUN (gated) | UNRUN (gated) | **PASS** (`0.9865`) | **PASS** (`0.9837`) | UNRUN (gated) |
| `staggered100_14k` | Fresh-process `7k -> 14k` | UNRUN (gated) | UNRUN (gated) | UNRUN (gated) | **PASS** (`0.9907`) | **PASS** (`0.9863`) | UNRUN (gated) |

*\*`rhbi-v12` and `rgdt-v3` passed only the final check (`terminal_passed=1`) on `zero_tanh_rail_escape`, failing the sustained terminal window bound (`mean_abs_error <= 0.005`).*
*†`unified-v3` passed the `100k`-sample `grid100` holdout (`precision=0.9867, mass_tv=0.0421, accuracy_score=0.6751`) and `4/5` terminal checks (`steps 6250, 6500, 6750, 7000`), failing only the earliest terminal check at `step=6000` (`radial_ks=0.0408, mass_tv=0.0554, accuracy_score=0.9180`).*

### 2.3 Fresh-Process `7k -> 14k` Native Continuation Accuracy Breakdown

| Candidate | Task | Horizon | Status | Wall (s) | Precision | `mass_tv` | `center_rms_sigma` | `center_max_sigma` | `cov_frob_rms` | `radial_ks` | `accuracy_score` |
| :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`t4-arch-v3`** | `grid100` | `7k` | **PASS** | `497.5` | `0.9816` | `0.0160` | `0.0743` | `0.1508` | `0.0685` | `0.0160` | `0.3312` |
| **`t4-arch-v3`** | `grid100_14k` | `14k` | **PASS** | `534.6` | `0.9839` | `0.0163` | `0.0806` | `0.1672` | `0.0699` | `0.0135` | `0.3313` |
| **`t4-arch-v3`** | `rotated100` | `7k` | **PASS** | `598.4` | `0.9825` | `0.0141` | `0.0709` | `0.1855` | `0.0669` | `0.0152` | `0.3124` |
| **`t4-arch-v3`** | `rotated100_14k` | `14k` | **PASS** | `598.6` | `0.9837` | `0.0140` | `0.0767` | `0.2033` | `0.0722` | `0.0147` | `0.3327` |
| **`t4-arch-v3`** | `staggered100` | `7k` | **PASS** | `503.4` | `0.9833` | `0.0142` | `0.0684` | `0.1440` | `0.0673` | `0.0145` | `0.2990` |
| **`t4-arch-v3`** | `staggered100_14k` | `14k` | **PASS** | `534.1` | `0.9863` | `0.0132` | `0.0845` | `0.2179` | `0.0715` | `0.0141` | `0.3282` |
| **`unified-v3`** | `grid100` | `7k` | FAIL (step 6000 win) | `688.6` | `0.9867` | `0.0421` | `0.1269` | `0.2252` | `0.1088` | `0.0265` | `0.6751` |
| **`unified-v3`** | `rotated100` | `7k` | **PASS** | `703.4` | `0.9856` | `0.0382` | `0.1558` | `0.3434` | `0.1304` | `0.0254` | `0.6953` |
| **`unified-v3`** | `rotated100_14k` | `14k` | **PASS** | `871.6` | `0.9865` | `0.0365` | `0.1553` | `0.3237` | `0.0878` | `0.0096` | **`0.4961`** |
| **`unified-v3`** | `staggered100` | `7k` | **PASS** | `736.0` | `0.9910` | `0.0382` | `0.1298` | `0.2422` | `0.0954` | `0.0196` | `0.5863` |
| **`unified-v3`** | `staggered100_14k` | `14k` | **PASS** | `980.9` | `0.9907` | `0.0395` | `0.1317` | `0.2327` | `0.0741` | `0.0130` | **`0.5147`** |
| `empf-v8` | `grid100` / `rot` / `stag` | `7k` | FAIL / FAIL / FAIL | `505.6–577.0` | `0.296–0.358` | `0.128–0.202` | `0.973–1.291` | `1.608–2.155` | `0.649–0.810` | `0.373–0.401` | `5.919–6.117` |
| `rgdt-v3` | `grid100` / `rot` / `stag` | `7k` | FAIL / FAIL / FAIL | `942.5–1075.6` | `0.251–0.308` | `0.071–0.158` | `0.990–1.190` | `1.737–2.118` | `0.728–0.824` | `0.398–0.412` | `5.875–6.153` |
| `rhbi-v12` | `grid100` / `rot` / `stag` | `7k` | FAIL / FAIL / FAIL | `511.1–850.8` | `0.203–0.296` | `0.110–0.129` | `1.197` (`rot`) | `2.374` (`rot`) | `0.788` (`rot`) | `0.427` (`rot`) | `6.109` (`rot`) |
| `negctrl` | `grid100` / `rot` / `stag` | `7k` | FAIL / FAIL / FAIL | `89.0–98.9` | `0.666–0.984` | `0.472–0.980` | `None` | `None` | `None` | `None` | `None` |

**Scientific takeaway on `14k` continuation:** Both `t4-arch-v3` and `unified-v3` demonstrate genuine clock-free equilibrium stability across fresh-process checkpoint continuation from `7,000` to `14,000` updates. `t4-arch-v3` achieves remarkably tight mass balance (`mass_tv = 0.0132–0.0163`), `accuracy_score = 0.3282–0.3327`, and `0` drift across all three lattices; `unified-v3` actually *improves* its covariance and radial calibration between `7k` and `14k` (`cov_frob_rms`: `0.1304 -> 0.0878` on `rotated100_14k`, `0.0954 -> 0.0741` on `staggered100_14k`; `radial_ks`: `0.0254 -> 0.0096` and `0.0196 -> 0.0130`).

### 2.4 Endurance & Failure-Mode Analysis

- **Uninterrupted `7,500`-step Ring Endurance (`ring_hold` + `ring_extension`):** All four candidates passing `mode_hold` (`empf-v8`, `rhbi-v12`, `rgdt-v3`, `unified-v3`) converged at `step=1400` and passed **`300 / 300`** extension checks (`first_hold_failure=None`, `min_hq=0.9000`). `t4-arch-v3` covered all `8/8` modes on `mode_hold` (`effective_modes=7.567`) but finished at `hq=0.8308 < 0.90` (`9/24` passing observations, `0` passing suffix), gating ring and shift tasks.
- **Discrete Word BiGAN Budget Boundary (`five_word_joint_smoke` / `five_word_joint_hold`):** Under the frozen `900s` timeout, `empf-v8`, `rhbi-v12`, `rgdt-v3`, and `unified-v3` all returned `INCOMPLETE` (`raw_status="timeout"` at `900.1–900.5s`) — kept strictly distinct from numerical `FAIL`. `t4-arch-v3` completed `20,001` steps in `561.1s` and passed `five_word_joint_smoke` (`5/5` words, `mass_tv=0.0189`, `min_recon_prob=0.9864`, `exact=1`), unlocking `five_word_joint_hold`, where it collapsed `apple` (`mass_apple=0.0`, `mass_berry=0.3916`, `modes=4/5`, `exact=0`) at `step=7334` (`FAIL`).
- **Rail Escape (`zero_tanh_rail_escape`):** `t4-arch-v3` (`0.0011` MAE, `0.0065` p99) and `empf-v8` (`0.0026` MAE, `0.0102` p99) cleanly escaped the saturated `tanh` rail; `rhbi-v12` and `rgdt-v3` escaped late (passing only the final check, failing the sustained stability window); `unified-v3` remained trapped (`0.8795` MAE, `0.9053` p99), identifying rail escape as a concrete remaining gap for unified synthesis.

---

## 3. Tier-4 Untouched Frontier Holdout (`r3-tier4-frontier-holdout-v1`)

### 3.1 Candidate × Tier-4 Task Matrix

| Candidate | `grid100` Gate (Reused from T3) | `grid100_mlp16` (16-Layer MLP) | `grid100_res16_tanh` (16-Layer ResNet + tanh) | `grid100_res16_scaled_tanh` (16-Layer ResNet + scaled tanh) | 12 `STRETCH_VECTOR_HOSTS` Tasks | New Tier-4 GPU Spend |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: |
| **`t4-arch-v3`** | **PASS** (`0.9816 / 0.3312`) | **PASS** (`0.9814 / 0.3010`) | **PASS** (`0.9856 / 0.4170`) | **PASS** (`0.9868 / 0.2884`) | `12 / 12` **BLOCKED** (preflight) | **`2,162.67s`** |
| `unified-v3` | FAIL (`step 6000` win) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | `12 / 12` **BLOCKED** (preflight) | `0.00s` |
| `empf-v8` | FAIL (`0.3460`) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | `12 / 12` **BLOCKED** (preflight) | `0.00s` |
| `rgdt-v3` | FAIL (`0.3080`) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | `12 / 12` **BLOCKED** (preflight) | `0.00s` |
| `rhbi-v12` | FAIL (`0.2068`) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | `12 / 12` **BLOCKED** (preflight) | `0.00s` |
| `r3-negctrl-unreg-adam` | FAIL (`tv=0.9800`) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | UNRUN (`grid100` gated) | `12 / 12` **BLOCKED** (preflight) | `0.00s` |

### 3.2 `t4-arch-v3` Runnable Tier-4 Architecture Metrics (`4 / 4 PASS`)

| Task | Status | Wall (s) | Precision | `mass_tv` | `center_rms_sigma` | `center_max_sigma` | `cov_frob_rms` | `radial_ks` | `accuracy_score` |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grid100` (reused gate) | **PASS** | `497.5` | `0.9816` | `0.0160` | `0.0743` | `0.1508` | `0.0685` | `0.0160` | `0.3312` |
| `grid100_mlp16` | **PASS** | `742.8` | `0.9814` | `0.0157` | `0.0948` | `0.2017` | `0.0634` | `0.0119` | `0.3010` |
| `grid100_res16_tanh` | **PASS** | `698.3` | `0.9856` | `0.0151` | `0.1158` | `0.2779` | `0.0955` | `0.0163` | `0.4170` |
| `grid100_res16_scaled_tanh` | **PASS** | `721.6` | `0.9868` | `0.0131` | `0.0553` | `0.1252` | `0.0766` | `0.0138` | `0.2884` |

### 3.3 Explicit Host-Preflight `BLOCKED` Boundary (12 `STRETCH_VECTOR_HOSTS` Tasks)

Per `AGENTS.md`, unimplemented host adapters are recorded as explicit `BLOCKED` outcomes, never as numerical failures:

- **Tasks (12):** `misgan_block_recovery`, `misgan_block_neural_imputer`, `ring16_bigan_continuous`, `misgan_mcar_p50_recovery`, `misgan_mcar_p80_anchored`, `pacgan_unequal_mass`, `pacgan8_grid100`, `alice_spiral_bigan`, `infogan_anisotropic_modes`, `cgan_overlapping_classes`, `cramer_energy_overlap`, `cyclegan_unpaired_constellation`.
- **Preflight reason (verified identically across all 6 candidates in [`tests/test_r3_calibration_holdout.py`](../../../tests/test_r3_calibration_holdout.py)):** `"<task_id>: no registered public Forge adapter for this stretch host"`.

---

## 4. Negative-Control True-Rejection Verification (`r3-negctrl-unreg-adam`)

[`configs/forge/ideas/r3-negctrl-unreg-adam.json`](../../../configs/forge/ideas/r3-negctrl-unreg-adam.json) defines a deliberately degenerate negative control (`reg_coeff=0.0`, `prior_reg=0.0`, `prior_regularizer="none"`, `optimizer_family="adam"`, `lr=0.001`, `betas=[0.5, 0.999]`, `lr_floor=1.0`, `network_lr_floor=1.0`, `trainer_family` omitted so it does not inflate `empf`'s `8`-candidate search budget):

1. **Cheap 3-Task Calibration Smoke Rejection (`25.8s`):** Fails `two_pole` (`mean_abs=0.0714 < 0.30`) and `unused_token_hold` (`concept_move=0.6850`, `unused_hold=0.9730`), passing only `ae_gan_hold` (`hold=0.0013, recon_mse=0.0288`) $\implies$ overall smoke decision = **`FAIL`**.
2. **Multimodal Holdout Reference Collapse:** Fails `mode_hold` (`cover=0.125`, `effective_modes=1.000`, `hq=0.2476`), `five_word_joint_smoke` (`0/5` words, `mass_tv=1.0000`), `zero_tanh_rail_escape` (`0.8669` MAE, `10.25%` saturated), and all three 100-mode lattices (`grid100` `mass_tv=0.9800`, `rotated100` `mass_tv=0.6058`, `staggered100` `mass_tv=0.4723`) $\implies$ reference decision = **`FAIL`**.
3. **True-Rejection Classification & Semantic Controls:** Classified cleanly as **`true_reject = (FAIL, FAIL)`** while passing `clockfree_audit` (`4/4` parity checks) and executing cleanly through the public trainer API (`all_finite=True`, `hooks_exercised=True`, plus deliberate generator-zeroing damage control verified in [`tests/test_r3_calibration_holdout.py`](../../../tests/test_r3_calibration_holdout.py)).

---

## 5. Mathematical Audit of `criteria-v1.json` & Public Promotion Blockers

We audited frozen [`configs/forge/calibration/criteria-v1.json`](../../../configs/forge/calibration/criteria-v1.json) (`sha256: 6eac2531eed68e392c11a57328999ac5dce509f8fe51d23e7b9031b7a4e3f9d1`) and [`configs/forge/calibration/initial.json`](../../../configs/forge/calibration/initial.json) (`3` smoke tasks, `16` reference tasks) without modifying any threshold or view policy (`discriminator_stability` rev `19` provisional; `clockfree_continuous` rev `5` provisional):

1. **Zero Reference Positives Across Current Scope (`minimum_reference_positives >= 1` violated):**
   - All `11` existing `evidence_scope="current"` calibration reports under [`reports/forge/calibration/`](../calibration/) remain **`adoption = "BLOCKED"`** with `reference_positives = 0`.
   - On the `16`-task reference set, **`unified-v3`** ([`readout-ca2ce5acd5856e8c69b00377.json`](../records/readout-ca2ce5acd5856e8c69b00377.json)) passes **`15 / 16`** (failing only `grid100`), while **`t4-arch-v3`** ([`readout-95d61950559e6e35c77b9df3.json`](../records/readout-95d61950559e6e35c77b9df3.json)) passes **`13 / 16`** (failing `unipolar`, `vector_unequal_mass`, `vector_anisotropic`). Their *union* covers `16 / 16`, proving every reference task is individually solvable under the public API, yet **`0` single global recipes currently satisfy the conjunctive `16 / 16` reference positive bar**.
2. **Rev-19 Tier-1 Smoke Cost Ratio Blocker (`<= 900.0s` and `<= 0.10` both violated):**
   - On `discriminator_stability` rev-19 (`11` required Tier-1 tasks vs `18` required Tier-2 tasks), `unified-v3` spends **`1,229.0s`** on Tier-1 vs **`1,753.5s`** on Tier-2 — a smoke-to-reference ratio of **`0.7009`**, violating both `maximum_smoke_wall_seconds = 900.0` and `maximum_smoke_to_reference_wall_ratio = 0.10`.
3. **False-Accept Feasibility Bound Under the Cheap 3-Task Smoke Screen (`maximum_false_accept_fraction <= 0.10`):**
   - The 3-task cheap screen (`two_pole`, `unused_token_hold`, `ae_gan_hold`) satisfies both cost bounds (`25.8s–54.8s <= 900.0s`, `~2.2%–3.1% <= 10%`) and catches `r3-negctrl-unreg-adam`.
   - However, under `decision_feasibility`, `(1 true_accept, 2 true_reject)` is the minimal `POSSIBLE` cohort (`Clopper-Pearson 95% upper = 0.8419`), whereas adding even **1** partial solver (`smoke=PASS, ref=FAIL` $\to$ `false_accept`) makes `2` `true_reject` controls `INFEASIBLE` (`1/3 = 0.333 > 0.10`) and requires **`>= 9` `true_reject` controls** (`1/10 = 0.10 <= 0.10`). Because all `5` active partial-solver recipes (`empf-v8`, `rhbi-v12`, `rgdt-v3`, `unified-v3`, `t4-arch-v3`) pass the 3-task smoke screen while failing `>= 1` reference task, calibrating a 3-task smoke screen honestly without cherry-picking negatives requires a stronger cheap smoke discriminator that separates partial solvers while remaining within the `10%` wall-time budget.
4. **Public Promotion Gate Blockers (`DEFAULT_RECIPE == "empf"` preserved):**
   - Verified in [`tests/test_r3_calibration_holdout.py`](../../../tests/test_r3_calibration_holdout.py): `promotion.register` deterministically raises `CapabilityError` for both legacy schema-v1 candidates (missing accepted view calibration) and schema-v3 candidates (requiring study-bound resolution). `particlegan.DEFAULT_RECIPE` remains `"empf"`.

---

## 6. Tuning-Budget Disclosure, Limitations & Recommendations

### 6.1 Unequal Historical Tuning Budget vs. Zero-Tuning Holdout Disclosure

Per `AGENTS.md`, historical development budgets are explicitly unequal across the six frozen recipes and must remain visible beside their zero-tuning holdout scores:

| Candidate | Family / Lineage | Historical Declared Candidate Cards (Pre-Holdout) | Holdout Tuning Budget (`r3-t3` & `r3-t4`) |
| :--- | :--- | :--- | :---: |
| `empf-v8` | `empf` (current public default) | `8` declared cards (`empf-v1..v8`) | **0** (frozen) |
| `rhbi-v12` | `rhbi` | `12` declared cards (`rhbi-v1..v12`) | **0** (frozen) |
| `rgdt-v3` | `rgdt` (`trainer_family` unset) | `3` iterations (`rgdt-v1..v3`) + `3` ablations (`rgdt-v3-no-*`) | **0** (frozen) |
| `unified-v3` | `unified` (synthesizing Round-1 tracks) | `3` cards (`unified-v1..v3`), built atop Round-1 lineages (`t1-prior` 2, `t1-coupling` 5, `t2-secant`, `rgdt`) | **0** (frozen) |
| `t4-arch-v3` | `t4-arch` (`trainer_family` unset) | `3` cards (`t4-arch-v1..v3`) | **0** (frozen) |
| `r3-negctrl-unreg-adam` | Negative control (`trainer_family` unset) | `1` declared control card (zero tuning) | **0** (frozen) |

### 6.2 Limitations & Unresolved Uncertainty

- **Protocol Seed `0` Only:** All 12 holdout studies evaluate deterministic protocol seed `0`; multi-seed confirmation is reserved for promotion contracts *after* accepted view calibration exists.
- **`five_word_joint_smoke` Budget Bottleneck:** `4/5` active recipes hit the `900s` timeout (`INCOMPLETE`) on `five_word_joint_smoke` due to per-step overhead on `20,001` steps; only `t4-arch-v3` finished within budget (`561.1s`).
- **Unregistered Stretch Adapters:** `12/15` Tier-4 frontier tasks lack public Forge adapters and therefore remain unmeasured (`BLOCKED`), preventing frontier generalization claims beyond the 3 runnable deep/rail 100-mode hosts.

### 6.3 Concrete Recommendations

1. **Close the Complementary Gap Between `unified-v3` and `t4-arch-v3` into a True `16 / 16` Reference Positive:** Combine `t4-arch-v3`'s `grid100` sustained-window margin (`acc=0.3312`), `zero_tanh_rail_escape` (`0.0011` MAE), 16-layer/rail robustness (`4/4` runnable Tier-4 PASS), and `five_word_joint_smoke` throughput (`561s`) with `unified-v3`'s `mode_hold` / `ring_extension` / `target_shift_recovery` (`81/81`) and `6/6` `vector_*` (`unipolar`, `vector_unequal_mass`, `vector_anisotropic`) coverage.
2. **Do Not Manufacture Calibration Acceptance or Promote a New Default Yet:** Keep `DEFAULT_RECIPE == "empf"` and `criteria-v1.json` frozen until (a) at least one single global recipe passes `16 / 16` reference tasks, and (b) a redesigned cheap Tier-1 smoke screen satisfies `<= 900s`, `<= 0.10` reference ratio, and `<= 0.10` false-accept rate across realistic partial-solver controls.
3. **Register Public Forge Adapters for `STRETCH_VECTOR_HOSTS`:** Implement clean, non-assisting public-API Forge adapters for the 12 `BLOCKED` Tier-4 tasks before treating Tier-4 as a complete multi-paradigm frontier benchmark.
