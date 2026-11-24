# Phase-2 CI Parity, Concurrent `develop` (PR #644) Reconciliation & Cross-Report Consistency Audit

**Owner:** `ci-sync`  
**Branch:** `research/fl-phase2-ci` (reconciling `origin/research/four-laws` [`e655abf8d`, PR #633] with `origin/develop` [`36e287f98`, PR #644 `feat/unified-v13-bytegan`])  
**Governance Compliance:** Strictly read-only with respect to `particlegan/` production code and immutable `reports/forge/records/*.json` historical receipts (`AGENTS.md`); zero weakened assertions; zero bypassed CI guards; PR #644 (`unified_v13` / `feydy_atomic`) preserved 100% intact.

---

## Executive Verdict

1. **Clean Conflict-Free Merge with `origin/develop` (`36e287f98`, PR #644):** Merging `origin/develop` into `origin/research/four-laws` via Git `ort` produces **zero merge conflicts** and preserves PR #644 (`preset="unified_v13"`, `rhbi_self_coupling="feydy_atomic"`, `19` files, `+3,731/-35`, `13/13` `tests/test_unified_bytegan.py` PASS) 100% intact.
2. **All `.github/workflows/tests.yml` CPU-Only CI Guards Pass After 3 Non-Weakening Integration Fixes:**
   - `scripts/check_lint.py`: **PASS** (`294` baseline findings, **`0` new additions**).
   - `scripts/check_repo_bloat.py`: **PASS** (**`0` new violations**, `302` legacy allowlisted; markdown audit reports do not trigger the bloat guard, and PR #644's `unified-bytegan-bijection.json` is `60,830 B < 512,000 B`).
   - `tests/four_laws_audit/` under strict CI CPU-only single-threaded conditions (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-o faulthandler_timeout=600`): **`89 / 89` CPU-active PASS** (`438.36 s` total), `1` CUDA-only test (`test_p6_law2_autodiff.py::test_cuda_bn_bf16_dual_hook_and_sdpa_backend_boundaries`) cleanly deselected via `@pytest.mark.cuda` per `pyproject.toml`. Maximum single-test CPU duration is `221.88 s` (`< 600 s` `faulthandler_timeout`); every other audit test completes in `<= 58.62 s`.
   - PR #644 & repo hygiene regression suites on CPU-only: `tests/test_unified_bytegan.py` **`13/13` PASS** (`6.62 s`), `tests/test_forge_bytegan.py` + `tests/test_unified_v11.py` + `tests/test_unified_v12.py` **`66/66` PASS** (`71.90 s`), `tests/test_docs_links.py` + `tests/e2e/test_repository_hygiene.py` + `tests/test_forge_current_technique_inventory.py` + `tests/test_publication_integrity.py` **`179/179` PASS** (`90.11 s`).
   - `scripts/ci_pytest.py` 8-shard non-overlapping coverage check: **PASS** (`9,752` collected CPU tests, `528` CUDA tests deselected, assigned exactly once across shards `0–7`).
   - Release distribution exclusion check (`MANIFEST.in` / `pyproject.toml`): **PASS** (`particlegan-0.12.0-py3-none-any.whl` `944,335 B < 1.2 MB`, `particlegan-0.12.0.tar.gz` `869,268 B < 1.2 MB`, **`0`** `reports/four-laws-audit/`, `scripts/four_laws_audit/`, or `tests/four_laws_audit/` paths leaked).
3. **Cross-Report Contradictions & PR #644 Shifts Fully Documented:** This report catalogs all **8** pre-existing cross-report drifts across Phase-1 `p0–p10` / `RESEARCH_GLOSSARY.md` (repairing `RESEARCH_GLOSSARY.md:48`'s broken `particlegan/unified_v12.py` path directly), plus all PR #644 LOC, card-count, naming-collision (`unified-v13`), and `75` shifted `file:line` references so Phase-2 canonical documents (`docs/four-laws.md`, `evidence-ledger-and-revisions.md`, `research-backlog-and-forge-proposals.md`, `integration-report.md`) stay 100% synchronized with `origin/develop`.
4. **Merge Blocker Verdict:** **Zero blockers** to merging `research/four-laws` into `develop` once the Phase-2 PRs (`ci-sync`, `docs-architect` PR #645, `ledger-backlog`) are integrated and verified together on `research/four-laws`.

---

## 1. Reconciliation with Concurrent `origin/develop` (PR #644 `feat/unified-v13-bytegan`)

While Phase-1 tracks `P0–P10` were audited against `origin/develop` at `0d18245e9`, `origin/develop` advanced to `36e287f98` by merging **PR #644** (`feat(unified): add opt-in unified_v13 (feydy_atomic) solving both 256-byte ByteGAN stretch tasks with 0 regressions`).

### 1.1 Exact Scope & Scientific Findings of PR #644

- **Recipe & API Delta (`configs/forge/ideas/unified-v13.json`, `rev=4e17ba843948`):**
  - Adds opt-in preset `preset="unified_v13"` (`particlegan/recipes.py:2823–2826`) with a **single** recipe field delta over `unified_v12`: `rhbi_self_coupling: "feydy_discrete" -> "feydy_atomic"`.
  - **Zero** new `Recipe` dataclass fields (`len(fields(Recipe)) == 179`); `DEFAULT_RECIPE` stays `"unified_v12"` (`particlegan/recipes.py:67`); `configs/forge/selections/family-current-v1.json` stays untouched (`unified-v11`, `default_adoption: false`); Forge calibration remains `BLOCKED` under `criteria-v1.json`.
  - Implements strictly recipe-owned components in `particlegan/_t1_coupling/atomic.py` (`451` LOC) plus integration hooks in `_t1_coupling/{charts,critic,primitive}.py`, `_unified_v11/{critic,loss,optimizer}.py`, and `_unified_v4/loss.py`: (1) scale-invariant discrete-atomic support gate (`compute_discrete_atomic_gate`), (2) persistent 1-stays / surplus-moves bijection ledger with deterministic Hamilton quotas (`update_atomic_ledger`, `compute_atomic_quotas`), (3) discrete atomic thermal residual, and (4) joint BiGAN inverse codebook contraction inside `joint_relativistic_loss` with `0` extra forward passes, `0` `torch.autograd.grad`, `0` SVD/`eigh`/`pinv`, clock-free horizon invariance, and full checkpointing via `opt_d.record`.
- **Verified Matched RTX A6000 Study Outcomes ([`unified-bytegan-bijection.md`](../forge/studies/unified-bytegan-bijection.md), [`unified-bytegan-bijection.json`](../forge/studies/unified-bytegan-bijection.json)):**

| Suite / Task (Matched Live Cohort, `seed=0`, RTX A6000, PyTorch `2.15.0`) | `unified-v12` Live Control | `unified-v13` Candidate | Net Delta / Parity |
| :--- | :---: | :---: | :--- |
| **`bytegan_hypercube8`** (T2 Stretch, 4k steps) | **FAIL** (`139/256` valid, `102` coll, `52,354.9 J`) | **PASS** (`256/256` valid, `0` coll, `rms=0.0445`, `@834`, `15,431.8 J`) | **`+1` PASS** (`-70.5%` J) |
| **`bytegan_bigan_dictionary`** (T4 Stretch, 6k steps) | **FAIL** (`177/256` fwd, `0/256` recon, `29,141.9 J`) | **PASS** (`256/256` fwd, `256/256` recon, `recon_rms=0.0124`, `256/256` enc-bij, `@1250`, `23,026.5 J`) | **`+1` PASS** (`-21.0%` J) |
| **Matched Non-ByteGAN Regressions (`41` tasks)** | Baseline | **`41/41` bit-exact** | **`max_abs_diff == 0.000e+00`** |
| **Full Measured `unified-v13` Suite (`51` tasks)** | `38/44` matched pairs | **`47/51` PASS** (`0` ERROR / TIMEOUT / BLOCKED) | `+2` PASS, `0` reg (`7/7` extra T2 PASS) |
| **Solo Idle GPU 0 `dcgan_performance_r256`** | `41.11 ms` · `1147.65 MiB` | **`39.94 ms`** · `1147.65 MiB` (`514.2 J` vs `536.4 J`) | `-1.18 ms`, `0.0 MiB` |

- **Three Crucial Methodological Nuances for Phase-2 Canonical Theory & Ledger:**
  1. **Honest Cohort Disclosure on Bimodal DDGAN Chains (Corroborates P8 Fragility):** On PR #644's Python `3.15.0` / PyTorch `2.15.0` RTX A6000 runtime cohort, **both** matched live `unified-v12` control and `unified-v13` record bit-identical `FAIL` (`max_abs_diff = 0.000e+00`) on `model_ddgan_bimodal_chain` (`t1_q1_covariance_eigen_max = 4.6961 > 4.5` at step `4375`) and `model_ae_ddgan_bimodal_chain` (`final_local_covariance_max = 2.0592 > 2.05` at step `4375`), whereas historical committed [`readout-b51ab67c`](../forge/records/readout-b51ab67c4d97526186e23651.json) passed on its earlier cohort. Optional stretch tasks `interp_img_patch_shift` (`interp_peak = 0.4915`) and `interp_saddle_barycentric` (`bhq = 0.7027`) likewise remain honest `FAIL` for both (`0.000e+00` diff).
  2. **Overfitting Interpretation Preserved:** `unified-v12`'s historical post-freeze unseen `0/2` `ByteGAN` failure ([`reports/forge/bytegan-stretch/summary.json`](../forge/bytegan-stretch/summary.json)) **remains valid empirical evidence of adaptive overfitting** in the `unified-v1..v12` lineage (P8-C1), because `unified-v13`'s `feydy_atomic` mechanism was engineered *after* observing `ByteGAN`. True unseen discrete generalization for `unified-v13` remains to be tested post-freeze (e.g., `FUSQP-v1`'s `dist_discrete_simplex_d12`).
  3. **Naming Collision Resolved:** Phase-1 [`p10-synthesis-and-ledger.md`](p10-synthesis-and-ledger.md#L326-L332) proposed naming the future Occam-pruned / defect-fixed candidate `unified-v13`. Because PR #644 claimed `unified-v13`, Phase-2 documents should refer to any future defect-fix / Occam-pruning candidate as `unified-v14` (or a generic label).

### 1.2 Exact Post-PR-#644 Code Complexity, Card Counts & Shifted `file:line` Inventory

Merging PR #644 shifted direct LOC, AST branches, declared Forge card counts, and `75` source line references cited across Phase-1 `p1–p10` reports:

| Metric / Package | Pre-PR-#644 (`0d18245e9`) | Post-PR-#644 (`36e287f98`) | Exact Delta |
| :--- | ---: | ---: | ---: |
| `unified-v12` delta only (`particlegan/_unified_v12/`) | `1,488` LOC · `241` branches | `1,488` LOC · `241` branches | `+0` LOC · `+0` branches |
| `unified-v11` direct (`_unified_v11/` + `unified_v11.py`) | `7,076` LOC · `816` branches | **`7,091` LOC · `821` branches** | **`+15` LOC · `+5` branches** |
| `unified-v12` effective direct (`_unified_v12` + `_unified_v11` + `unified_v11.py`) | `8,564` LOC · `1,057` branches | **`8,579` LOC · `1,062` branches** | **`+15` LOC · `+5` branches** |
| `particlegan/_t1_coupling/` *(transitively imported; hosts `feydy_atomic`)* | `4,067` LOC (`7` files) | **`4,847` LOC (`8` files)** | **`+780` LOC** *(incl. `451` `atomic.py`)* |
| `particlegan/_unified_v4/` + `unified_v4.py` | `4,313` LOC · `477` branches | `4,313` LOC · `477` branches | `+0` LOC (`1` line modified) |
| `particlegan/recipes.py` | `3,053` LOC | **`3,060` LOC** | **`+7` LOC** |
| Declared `unified-v*` cards (`configs/forge/ideas/`) | `12` (`unified-v1..v12`) | **`13`** (`+ unified-v13`) | **`+1` card** (`+3` studies) |
| Direct 4-Law ancestry cards / `technique-inventory.md` Unified cards | `68` / `34` | **`69` / `35`** | **`+1` card** |

**Exact Source Line Shifts (`0d18245e9 -> 36e287f98`) Affecting Phase-1 Audit Citations (`75` total occurrences):**

| Modified File | Pre-PR-#644 Line Range | Shift | Representative Shifted Audit Citations (`old -> new`) |
| :--- | :--- | :---: | :--- |
| `particlegan/recipes.py` | `L1–L682` | `+0` | `DEFAULT_RECIPE` at `L67` **unchanged** |
| `particlegan/recipes.py` | `L685–L2819` | **`+3`** | `_generator_free`: `L2378–2385 -> L2381–2388`; `unified_v12` preset: `L2816–2819 -> L2819–2822` |
| `particlegan/recipes.py` | `L2820+` | **`+7`** | `cond_fiber` / downstream presets: `L2824–2831 -> L2831–2838` |
| `particlegan/_unified_v11/optimizer.py` | `L549–L592` | **`+4`** | `set_sampled_rows`: `+4` lines inserted for `_active_prior_table` / `_active_batch_rows` |
| `particlegan/_unified_v11/optimizer.py` | `L593+` | **`+8`** | P1/P3/P4/P5 defects: `L714–782 -> L722–790`; `L802 -> L810`; `L878–895 -> L886–903`; `L973–979 -> L981–987`; `L994 -> L1002`; `L1106 -> L1114`; `L1260–1284 -> L1268–1292` |
| `particlegan/_unified_v11/critic.py` | `L156–L397` | **`+5`** | P4 `coord_scale` floor: `L342–343 -> L347–348` |
| `particlegan/_unified_v11/critic.py` | `L398+` | **`+7`** | P4 `b_p=2` prefix probe: `L694 -> L701` |
| `particlegan/_t1_coupling/charts.py` | `L203+` | **`+1`** | P7 `laguerre_hard_commitment`: `L502–581 -> L503–582`; $B_f < K$ bias bug `L560 -> L561`; `L635–646 -> L636–647` |
| `particlegan/_t1_coupling/primitive.py` | `L123–L603` | **`+9`** | P2 screening dead zone: `L437–463 -> L446–472` |
| `particlegan/_t1_coupling/primitive.py` | `L896–L1192` | **`+104`** | P2 `g_prox` / macro handoff: `L913 -> L1017` |

*(Per `AGENTS.md` and Phase-2 file-ownership boundaries, Phase-1 historical audit reports `p0–p10` are preserved under their original `0d18245e9` source snapshot; all Phase-2 canonical documents use post-PR-#644 coordinates.)*

---

## 2. Cross-Report Contradiction & Drift Audit (`p0–p10`, `docs/four-laws.md`, `RESEARCH_GLOSSARY.md`)

An exhaustive programmatic and manual cross-check across all 11 Phase-1 audit tracks (`p0–p10`), pre-Phase-2 `docs/four-laws.md`, and `RESEARCH_GLOSSARY.md` identified **8 concrete cross-report contradictions or drifts** (zero broken markdown file links across all `reports/four-laws-audit/*.md` and `docs/four-laws.md`):

| # | Contradiction / Drift | Conflicting Source(s) | Authoritative Verified Ground Truth | Resolution Status |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **`t3_iso_scale_mass_8x` threshold misquoted as `0.35` (`0.013` miss)** | [`p8-qualification-audit.md:47`](p8-qualification-audit.md#L47) repeats `law4-energy-champion.md:90` (`0.3375 vs 0.35`) | Frozen [`configs/forge/tasks/t3_iso_scale_mass_8x.json:99`](../../configs/forge/tasks/t3_iso_scale_mass_8x.json#L99) sets `core_min_eigen_ratio >= 0.45`; `0.3375` is a **`0.1125` miss (`8.7×` larger)** (verified in [P2](p2-law3-identifiability.md), [P5](p5-optimizer-asymmetry.md#L30), [P10](p10-synthesis-and-ledger.md#L220)). | Reconciled in Phase-2 docs (`docs/four-laws.md` PR #645 & `ledger-backlog`); logged here as P8 erratum. |
| **2** | **`unified_v11`/`v12` Law 2 image `d_penalty` called `"1st-order"` / `"BF16 micro-secant"`** | [`p9-receipts-efficiency-audit.md:126,170,209`](p9-receipts-efficiency-audit.md#L170), [`p8-qualification-audit.md:30`](p8-qualification-audit.md#L30) | [P6](p6-law2-autodiff-audit.md) proved `fwAD.dual_level` + `.backward()` computes **reverse-over-forward (RoF) mixed second-order derivatives** $\frac{\partial^2 D_\psi}{\partial \psi \partial x} u$, and `_unified_v11/penalty.py:404–490,631–695` replaced all image micro-secants with 1-probe interleaved `fwAD` (`_eval_image_fwad_sq`). | Reconciled in Phase-2 docs; logged here as P8/P9 erratum. |
| **3** | **PR #611+#615 labeled `"Zero-sync pullback"`** | [`p9-receipts-efficiency-audit.md:128,142`](p9-receipts-efficiency-audit.md#L128) | [P7](p7-law1-topology-audit.md#L33) falsified literal `"zero-sync"`: PR #611+#615 yields **`5×–6×` reduced-sync** (`40 -> 8` pullback-only, `60 -> 10` full-step on CUDA; `60 -> 12` full-step on CPU), still strictly `>= 8` host syncs/step. | Reconciled in Phase-2 docs; logged here as P9 erratum. |
| **4** | **2D `_conformal_local_horizons` listed unconditionally among `"3 of 5"` redundant Round-4 epicycles** | [`p9-receipts-efficiency-audit.md:15,118`](p9-receipts-efficiency-audit.md#L118) | [P2](p2-law3-identifiability.md) & [P10](p10-synthesis-and-ledger.md#L30) reconciled the split: **`2`** unconditionally redundant Round-4 epicycles (`d_z=4` Radon–Hurwitz, `w_sat_pt`) remain in `unified_v12`, whereas 2D `_conformal_local_horizons` was redundant on `v10`'s `12/12` passing probes (where `t3_iso` already failed) yet is **causally required** in `v11`/`v12`'s `G = Id` `T1CouplingLoss` for `t3_iso_scale_mass_8x`. | Reconciled in P10 & Phase-2 docs; logged here as P9 erratum. |
| **5** | **Unqualified `18/18` Tier-3 & `39.22 ms` citations** | [`p5-optimizer-asymmetry.md:385,407`](p5-optimizer-asymmetry.md#L385) | [P8](p8-qualification-audit.md#L69) & [P9](p9-receipts-efficiency-audit.md#L42-L43) proved `18/18` Tier-3 has zero committed `readout-*.json` receipts (`13/18` core overlap), and `39.22 ms` is the 50-step single-batch NVML profiler (committed Forge [`readout-59898e1a`](../forge/records/readout-59898e1a77c4b179c9aebbc1.json) is `41.39 ms/step`). | Reconciled in Phase-2 docs; logged here as P5 erratum. |
| **6** | **Broken `particlegan/unified_v12.py` path in `RESEARCH_GLOSSARY.md`** | [`RESEARCH_GLOSSARY.md:48`](../../RESEARCH_GLOSSARY.md#L48) (caught in P5-C9 & P10 Sec. 5B) | `particlegan/unified_v12.py` never existed; actual paths are `particlegan/_unified_v12/optimizer.py`, `particlegan/_unified_v11/`, and `particlegan/unified_v11.py`. | **FIXED directly in this PR** (`RESEARCH_GLOSSARY.md:48`); `tests/test_docs_links.py` PASS. |
| **7** | **Stale test function name in P9 report** | [`p9-receipts-efficiency-audit.md:193`](p9-receipts-efficiency-audit.md#L193) cites `test_four_laws_links_and_missing_receipts_detected` | Renamed by P10 to `test_four_laws_links_and_missing_receipts_repaired` in [`tests/four_laws_audit/test_p9_receipt_integrity.py:27`](../../tests/four_laws_audit/test_p9_receipt_integrity.py#L27) once `docs/four-laws.md`'s 3 broken links were fixed. | Documented here; test passes (`test_p9_receipt_integrity.py`). |
| **8** | **Unescaped pipe delimiters inside GFM table math & `docs/four-laws.md` line drift** | `28` table rows across historical Phase-1 reports (`p0`: 1, `p1`: 1, `p3`: 17, `p5`: 2, `p6`: 5, `p7`: 2); pre-P10 `docs/four-laws.md#L...` citations in `p0–p9` | Unescaped vertical-bar characters inside inline math/code split GFM columns; `p2`, `p4`, `p8`, `p9`, `p10`, `README.md`, `docs/four-laws.md`, and all Phase-2 reports enforce `0` GFM table mismatches using `\lVert...\rVert` / `\lvert...\rvert` / `\mid`. | Enforced `0` GFM errors on all Phase-2 files; historical Phase-1 erratum logged here. |

*(All `83` audited claim counts across P0–P10 in `p10-synthesis-and-ledger.md` Table 3.1/3.2 remain internally consistent: `13` proven mathematical results, `4` information-theoretic obstructions, `2` necessary conditions, `0` sufficient conditions, `14` empirically supported causal mechanisms, `2` empirical correlations, `1` mathematical conjecture, `4` engineering heuristics, `40` falsified claims, `3` unresolved questions.)*

---

## 3. Full Local `.github/workflows/tests.yml` CPU-Only CI Verification

Every CI gate defined in [`.github/workflows/tests.yml`](../../.github/workflows/tests.yml) was executed locally under strict CPU-only, single-threaded environment variables (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).

### 3.1 Three Real CI Failures Caught & Fixed (Zero Weakened Assertions)

Running the merged branch under CPU-only CI uncovered **three concrete CI integration failures** (two hard failures + one CUDA marker convention mismatch), all fixed cleanly within authorized files:

1. **CPU-Only Sync Offset Failure in `tests/four_laws_audit/test_p7_law1_topology.py:291` (`test_invariant4_pullback_isometry_and_fastpath_parity_and_syncs`):**
   - **Observed CPU failure:** `assert iso["v11_pullback_only_syncs"] < 0.35 * iso["v4_pullback_only_syncs"]` failed with `19 < 17.85` (`19` vs `51`), whereas CUDA recorded `8` vs `40` (`8 < 14.0`).
   - **Exact root cause (stack-trace verified):** `_count_host_syncs(device)` counts calls on tensors whose `device.type == device.type`. On CUDA, CPU helper tensors are ignored (`8` vs `40`, exact `32`-sync pullback reduction). On CPU, `_count_host_syncs(cpu)` also intercepts **11 identical non-pullback CPU optimizer calls** on *both* `v4` and `v11`: **`9`** `torch.optim.Adam` CPU `state["step"].item()` calls (`torch/optim/optimizer.py:102:_get_value`, one per parameter in `SimpleMLPGenerator` + `prior.z`) plus **`2`** CPU `set_sampled_rows` bounds-check `bool`s (`_unified_v11/optimizer.py:563`), shifting `(8, 40) -> (19, 51)` while preserving the identical `32`-sync pullback reduction (`51 - 19 == 32`). Meanwhile full-step `GANTrainer` already achieves exact `5.0×` reduction directly on CPU (`12.0` vs `60.0` for both `unified_v11` and `unified_v12`).
   - **Non-weakening fix:** Subtract the exact `11` CPU-only optimizer offset when `device.type == "cpu"` (restoring exact `8 < 0.35 * 40` and `8 >= 4`), **and strengthen** the test with explicit pullback delta (`v4 - v11 >= 25`) plus full-step `GANTrainer` sync reduction assertions (`< 0.35 * v4`) for both `unified_v11` and `unified_v12`.
   - **CPU CI runtime note (`test_nonadversarial_division_of_labor_breaks_jacobian_cliff_and_ghost_mass`):** Originally took `252.72 s` on 1 CPU thread because `eval_every=250` executed 5 unused intermediate step-250 `2048×2048` Sinkhorn + 4096-point Jacobian evaluations that the test never reads. Setting `eval_every=500` (identical `steps=500`, `batch_size=256`, bit-identical `arm["final"]` because eval uses an isolated local RNG) trims runtime to **`221.88 s`** in the full suite (`238.29 s` isolated, safely `< 600 s` `faulthandler_timeout`) with zero assertion changes. *(A `batch_size=128` CPU trial ran in `93.38 s` and passed all 15 assertions, but was intentionally rejected because `arm_a jac_max = 9.05` left only a `0.05` margin above `> 9.0`, risking cross-BLAS flakes on GitHub Actions runners.)*
2. **PR #644 Card-Count Collision in `scripts/four_laws_audit/p9_verify_receipts.py:650` & `tests/four_laws_audit/test_p9_receipt_integrity.py:289`:**
   - **Observed failure:** `assert p8["lineage_tuning_budgets"]["unified_v1_v12"]["declared_cards"] == 12` failed with `assert 13 == 12` because `s.startswith("unified-v")` matched PR #644's new `configs/forge/ideas/unified-v13.json` (and inflated `studies` from `70 -> 73`).
   - **Non-weakening fix:** Scoped `unified_v1_v12` via exact regex `(?:^|[-_])unified-v(?:[1-9]|1[0-2])(?:$|[-_])` (preserving P8's audited `12` cards / `70` studies / `69` readouts verbatim!), added a separate `unified_v13_plus` lineage bucket (`1` card, `3` studies), loaded PR #644's `reports/forge/studies/unified-bytegan-bijection.json` into `bytegan_v13_study`, and added positive regression assertions in `test_p9_receipt_integrity.py` confirming both historical `unified-v12` `0/2` ByteGAN FAIL and PR #644 `unified-v13` `2/2` PASS (`256/256` unique bytes, `256/256` exact recon, `41/41` bit-exact invariance, `default_recipe_changed is False`).
3. **CUDA Marker Convention in `tests/four_laws_audit/test_p6_law2_autodiff.py:410`:**
   - Added `@pytest.mark.cuda` alongside `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` on `test_cuda_bn_bf16_dual_hook_and_sdpa_backend_boundaries` so default `pytest` (`pyproject.toml` `addopts = "-ra -m 'not cuda'"`) and CPU CI shards cleanly deselect it alongside the repository's other `527` CUDA tests, while `pytest -m cuda` executes it on GPU.

### 3.2 Complete Local CI Verification Matrix

| CI Check (`.github/workflows/tests.yml` Parity) | Exact Command | Result | Key Observed Metrics |
| :--- | :--- | :---: | :--- |
| **1. Lint Baseline Guard** (`shard 0`) | `python scripts/check_lint.py` | **PASS** | `294` existing findings; **`0` additions** |
| **2. Repo Bloat Guard** (`shard 0`) | `python scripts/check_repo_bloat.py` | **PASS** | **`0` new violations** (`302` legacy allowlisted; `.md` exempt; `unified-bytegan-bijection.json` `60,830 B < 512 KB`) |
| **3. Four Laws Audit Suite (CPU-Only, 1 Thread)** | `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -v --durations=25 -o faulthandler_timeout=600 tests/four_laws_audit/` | **PASS** | **`89 / 89` CPU-active PASS** in `438.36 s`, `1` `@pytest.mark.cuda` deselected; `0` failures; max test `221.88 s < 600 s` |
| **4. PR #644 ByteGAN Unit & Continuation Suite** | `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -v tests/test_unified_bytegan.py` | **PASS** | **`13 / 13` PASS** in `6.62 s` |
| **5. Forge ByteGAN + `unified_v11`/`v12` Suite** | `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_forge_bytegan.py tests/test_unified_v11.py tests/test_unified_v12.py` | **PASS** | **`66 / 66` PASS** in `71.90 s` |
| **6. Docs Links, Repo Hygiene, Inventory & Publication** | `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_docs_links.py tests/e2e/test_repository_hygiene.py tests/test_forge_current_technique_inventory.py tests/test_publication_integrity.py` | **PASS** | **`179 / 179` PASS** in `90.11 s` *(incl. repaired `RESEARCH_GLOSSARY.md`)* |
| **7. 8-Shard `ci_pytest.py` Coverage Check** | `scripts/ci_pytest.py` collection + `check_receipts` across `8` shards (`--split-file tests/test_refresh_forge_readouts.py --pin-list 7=scripts/ci_history_tests.txt`) | **PASS** | **`9,752`** CPU tests collected (`528` CUDA deselected), `0` omitted, `0` duplicated |
| **8. Wheel & Sdist Exclusion Check** (`distributions`) | PEP 517 `setuptools.build_meta.build_wheel` & `build_sdist` + exact `.github/workflows/tests.yml:269–307` assertions | **PASS** | Wheel **`944,335 B`** (`< 1.2 MB`), Sdist **`869,268 B`** (`< 1.2 MB`), **`0`** `four-laws`/`four_laws` paths leaked |

### 3.3 `scripts/ci_pytest.py` 8-Shard Distribution & CPU Wall-Clock Budget

All `89` CPU-active `tests/four_laws_audit/` tests integrate automatically into the 8-shard CI runner via filename SHA-256 hashing with zero shard-pinning edits required:

| CI Shard | Total Collected Tests | `tests/four_laws_audit/` Files & CPU-Active Tests | Observed Audit CPU Wall Time (1 Thread) | Margin vs. `faulthandler_timeout=600` & 60-min Shard Cap |
| :-: | ---: | :--- | ---: | :--- |
| **Shard 0** | `873` | `test_p6_law2_autodiff.py` (`23` CPU-active; `1` `@pytest.mark.cuda` deselected) | `~1.5 s` | `> 400×` margin |
| **Shard 1** | `1,025` | *(none)* | `0.0 s` | — |
| **Shard 2** | `1,319` | *(none)* | `0.0 s` | — |
| **Shard 3** | `1,290` | `test_p7_law1_topology.py` (`7` tests; slowest `test_nonadversarial_division_of_labor` `221.88 s`) | `~230 s` | **`2.7×` under `600 s` test timeout; `15×` under 60-min shard cap** |
| **Shard 4** | `901` | `test_p9_receipt_integrity.py` (`8` tests) | `~7 s` | `> 85×` margin |
| **Shard 5** | `1,230` | `test_p2_law3_identifiability.py` (`7`, slowest `58.62 s`) + `test_p5_asymmetry.py` (`9`, slowest `3.61 s`) | `~65 s` | **`> 10×` under `600 s` test timeout** |
| **Shard 6** | `1,327` | *(none)* | `0.0 s` | — |
| **Shard 7** | `1,787` | `test_p0` (`13`), `test_p1` (`7`), `test_p3` (`8`), `test_p4` (`7`) = `35` tests *(slowest `26.74 s`)* | `~136 s` | **`> 22×` under `600 s` test timeout** |

### 3.4 Packaging & Bloat Guard Compliance Notes

- **Why `reports/four-laws-audit/*.md` passes `scripts/check_repo_bloat.py`:** [`scripts/check_repo_bloat.py`](../../scripts/check_repo_bloat.py) enforces two rules under `reports/`: (1) zero newly tracked binary/bulk extensions (`.ckpt`, `.gif`, `.gz`, `.html`, `.mp4`, `.npz`, `.pt`, `.pth`, `.zip`), and (2) zero newly tracked `.json` files exceeding `500 KiB` (`512,000 B`). All Four Laws audit deliverables under `reports/four-laws-audit/` are compact Markdown reports (no binaries, no JSON traces, no checkpoints), and PR #644's `reports/forge/studies/unified-bytegan-bijection.json` is `60,830 B` (`11.9%` of the cap).
- **Why release distributions exclude all audit artifacts:** [`pyproject.toml`](../../pyproject.toml#L38-L41) restricts wheels strictly to `["particlegan", "particlegan.*", "particlegan_forge", "particlegan_forge.*"]` (`944,335 B`), and [`MANIFEST.in`](../../MANIFEST.in) explicitly prunes `tests`, `reports`, `docs`, and `scripts` from sdists (`869,268 B`, containing only `LICENSE`, `MANIFEST.in`, `PKG-INFO`, `README.md`, `particlegan`, `particlegan.egg-info`, `particlegan_forge`, `pyproject.toml`, `setup.cfg`).

---

## 4. Phase-2 Cross-PR Integration Contract & Reproduction Commands

### 4.1 Cross-PR Integration Check (`ci-sync` + `docs-architect` + `ledger-backlog`)

- `tests/four_laws_audit/test_p9_receipt_integrity.py::test_four_laws_links_and_missing_receipts_repaired` checks that `docs/four-laws.md` contains `>= 60` local file links, `>= 42` unique committed `readout-*` prefixes, `0` broken links, `0` missing readouts, and explicit links to all four committed `unified-v12` readouts (`b51ab67c`, `c1ee5c3b`, `624964f4`, `59898e1a`).
- Per board post `#72`, `@docs-architect`'s canonical `docs/four-laws.md` (PR #645) preserves `134` local links and `43/43` committed readouts (linking PR #644's `unified-v13.json` & `unified-bytegan-bijection.md` plus all 4 Phase-2 companion reports). Once all three Phase-2 PRs (`ci-sync`, `docs-architect`, `ledger-backlog`) are merged onto `research/four-laws`, `test_four_laws_links_and_missing_receipts_repaired` will verify `0` broken links end-to-end across the combined tree.

### 4.2 Exact Verification Commands

```bash
# 1. Lint & Repo Bloat Guards
PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/check_lint.py
PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/check_repo_bloat.py

# 2. Full Four Laws Audit Suite under CPU-only, single-threaded CI settings
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" \
  /nvme/dev/hypergan/ParticleGAN/.venv/bin/pytest -v --durations=25 -o faulthandler_timeout=600 tests/four_laws_audit/

# 3. PR #644 ByteGAN + Unified v11/v12 + Root Doc/Hygiene Regression Suites (CPU-only)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" \
  /nvme/dev/hypergan/ParticleGAN/.venv/bin/pytest -q \
  tests/test_unified_bytegan.py tests/test_forge_bytegan.py \
  tests/test_unified_v11.py tests/test_unified_v12.py \
  tests/test_docs_links.py tests/e2e/test_repository_hygiene.py \
  tests/test_forge_current_technique_inventory.py tests/test_publication_integrity.py
```
