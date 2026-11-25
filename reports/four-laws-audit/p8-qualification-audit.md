# P8 Audit: Scientific Qualification, Forge Overfitting, Calibration Governance, and Frozen Unseen Qualification Protocol

**Audit Track:** P8 — Scientific Qualification & Benchmark Overfitting  
**Integration Branch:** `research/four-laws` (branched from `origin/develop` at `0d18245e9`)  
**Audited Documents & Artifacts:** [`docs/four-laws.md`](../../docs/four-laws.md), [`reports/forge/studies/r3-calibration-holdout.md`](../forge/studies/r3-calibration-holdout.md), [`configs/forge/calibration/criteria-v1.json`](../../configs/forge/calibration/criteria-v1.json), [`reports/forge/calibration/r3-calibration-holdout.json`](../forge/calibration/r3-calibration-holdout.json), [`configs/forge/selections/family-current-v1.json`](../../configs/forge/selections/family-current-v1.json), [`reports/forge/technique-inventory.md`](../forge/technique-inventory.md), [`reports/forge/problems.md`](../forge/problems.md), [`reports/forge/bytegan-stretch/summary.json`](../forge/bytegan-stretch/summary.json), [`reports/forge/records/readout-*.json`](../forge/records/), and [`reports/forge/technique-receipts/*.json`](../forge/technique-receipts/).  
**Reproducible Verification Code:** [`scripts/four_laws_audit/p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) and [`tests/four_laws_audit/test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py).

---

## Executive Verdict

1. **Adaptive Research-Level Overfitting Is Real and Already Empirically Demonstrated.** Although ParticleGAN enforces "one unchanged global recipe across tasks" and bans seed-only score shopping, the lineage culminating in `unified-v12` accumulated **12 `unified-v*` cards (`69` readouts, `683` task evaluations, `21.59` GPU-h)**, **11 `law4-energy-v*` cards (`27` readouts, `366` evaluations, `17.69` GPU-h)**, plus **45 directly upstream Round 1–4 cards (`94` readouts, `1,094` evaluations, `38.99` GPU-h)** — **68 declared cards, `190` concluded readouts, `2,143` task evaluations, and `78.27` GPU-hours** iterating against a **fixed seed-0 benchmark suite**. When `unified-v12` was tested *after* freezing on two genuinely unseen discrete tasks (**ByteGAN**, PR #632), it scored **`0/2` FAIL** (`139/256` valid unique bytes on `bytegan_hypercube8`; `170/256` unique bytes and `0/256` exact reconstructions on `bytegan_bigan_dictionary`), and when transferred to a different GPU/host cohort (**NVIDIA L4** Colab waves 7a/7b), it **FAILED** on `grid100_res16_tanh`, `rotated100_affine_square_named_v1`, and `vector_unequal_width_published`, while timing out twice on `model_ae_ddgan_gaussian_chain` and erroring on two `grid100_*` variants.
2. **Honest Calibration Disclosure Coexists With Governance Drift.** [`docs/four-laws.md`](../../docs/four-laws.md#L206-L221) honestly discloses that screening calibration remains `BLOCKED` under frozen [`configs/forge/calibration/criteria-v1.json`](../../configs/forge/calibration/criteria-v1.json) and that `default_adoption` is `false`. However, the repository violated its own published stop rule ([`r3-calibration-holdout.md`](../forge/studies/r3-calibration-holdout.md#L200): *"Keep `DEFAULT_RECIPE == 'empf'` until accepted calibration"*) twice: promoting `DEFAULT_RECIPE` to `unified_v11` (`0.11.0`) and then `unified_v12` (`0.12.0`, [`particlegan/recipes.py`](../../particlegan/recipes.py#L67)) while [`configs/forge/selections/family-current-v1.json`](../../configs/forge/selections/family-current-v1.json#L1063-L1081) still pins `unified-v11` with `"default_adoption": false`, and while `docs/four-laws.md` omits `unified-v12`'s genuine failures (e.g., `interp_img_patch_shift` `interp_peak=0.4915` ghosting on RTX A6000) and cites uncommitted Tier-3 holdout (`18/18`) and shift-recovery (`5/5`) passes without committed `readout-*.json` receipts.
3. **Operational Forge Gate != Mathematical Law.** Passing a finite seed-0 operational threshold establishes at best an *existential empirical witness* on that task/architecture/seed/GPU tuple. It never proves mathematical necessity, sufficiency, or invariance — a gap underscored by the 1-ULP step-6250 `modes=99` cliff on `grid100_res16_tanh`, by special-cased dimension branches (`d_z == 2`, `d_z == 4`), and by architecture-transfer failures (`vector_unequal_width_published`, `rotated100_affine_square_named_v1`).
4. **Remedy:** Section 5 specifies **`FUSQP-v1`** (Frozen Unseen Scientific Qualification Protocol), a strictly post-freeze, distribution-level, architecture/invariant/distribution-disjoint confirmatory suite that honors [`AGENTS.md`](../../AGENTS.md)'s ban on seed-only tuning while ending single-seed screening conflation.

---

## 1. Reconstructed Tuning Budget & Adaptive Selection Bias

### 1.1 Complete Quantitative Lineage Accounting

Programmatically extracted from `configs/forge/{ideas,configurations,studies}/*.json` and `reports/forge/records/readout-*.json` via [`scripts/four_laws_audit/p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py):

| Lineage / Development Track | Declared Cards | Registered Studies | Concluded Readouts | Task Evaluations | Recorded Wall Time (GPU-h) | Role in `unified-v12` Ancestry |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **`unified_v1_v12`** (`unified-v1..v12`) | **12** | **70** | **69** | **683** | **21.59** | Direct unified synthesis & tuning |
| **`law4_energy_opt`** (`law4-energy-v1..v11`) | **11** | **29** | **27** | **366** | **17.69** | Direct optimizer precursor to `unified-v12` |
| `t1_prior_coupling` (`t1-prior-v1..2`, `t1-coupling-v1..5`) | 7 | 14 | 14 | 208 | 8.76 | Law 1 prior atlas, LTV, Laguerre coupling |
| `t2_secant` (`t2-secant-v1..5`) | 5 | 8 | 8 | 123 | 2.78 | Law 2 1st-order `fwAD`/secant critic |
| `t3_rgdt` (`rgdt-v1..3` + ablations) | 6 | 18 | 18 | 228 | 7.54 | Law 3 scale-invariant $k=4$ / Weibull AKT |
| `t4_arch_nash` (`t4-arch-v1..3`, `t4-nash-v1..3`, controls) | 15 | 26 | 26 | 234 | 7.95 | Law 4 dynamic-$`10^{-38}`$ / NS / `delta_g_max` |
| `cond_fiber` (`cond-fiber-v1`) | 1 | 2 | 2 | 53 | 1.16 | Law 3 conditional fiber bank |
| `r4_razor_minimal` (`r4-razor-*`, `minimal-v1..2`) | 11 | 30 | 26 | 248 | 10.80 | Round-4 ablation & minimal-core diagnostics |
| **Subtotal — Direct 4-Law / Unified Ancestry** | **68** | **197** | **190** | **2,143** | **78.27** | *Excludes `law4-sgd-*`/`law4-alt-*` branch runs* |
| *Prior historical baselines (`empf`, `rhbi`, `bispf`, `bcap`)* | *514* | *175* | *115* | *1,370* | *30.70* | Historical development & comparator pool |
| **Total Repository Catalog (all families)** | **674** *(194 ideas + 480 configs)* | **501** | **453** | — | — | Plus `133` searches & `3,789` technique-receipts |

> **Note on Hidden Off-Forge Budget:** The table above counts *only* committed Forge readout records. It excludes: (a) the `law4-sgd-anatomy` and `law4-alt-optimizers` companion branches (`12` optimizer arms, `16` readouts cited in [`law4-optimizer-laws.md`](../forge/studies/law4-optimizer-laws.md)), (b) uncommitted Phase-3/Phase-4 local queue runs (such as the uncommitted `3bf87cea3+ff41cf173` Tier-3 holdout, `t4-arch`, and `dcgan_performance` runs), and (c) local iterative debugging cycles between committed revisions (e.g., `unified-v1` through `unified-v12` each represent an entire PR/debugging cycle whose intermediate failures are not recorded as Forge readouts).

### 1.2 Mechanics of Adaptive Overfitting Under a "Single Global Recipe"

A subtle methodological hazard in [`docs/four-laws.md`](../../docs/four-laws.md) is treating **"one unchanged global recipe across tasks"** plus **"no seed-only search"** as sufficient protection against benchmark overfitting. In adaptive data analysis (Dwork et al., 2015; Gelman & Loken, 2014), repeated human/agent iteration against a fixed finite test set induces severe selection bias through **branching mechanism accumulation** even when every candidate is a single global recipe on seed `0`:

1. **Threshold-Driven Mechanism Patching ("Whack-a-Mole" Epicycles):** Every single version jump from `unified-v1` through `unified-v12` was explicitly engineered to convert specific failing Forge seed-0 tasks into `PASS`:
   - `unified-v10 -> v11`: added `zero_eigh` CondFiber bank for `model_ddgan_gaussian_chain`, `direction_blend` for `residual_student`, $k=4$ ratio gate (`3.52`, `1.30`) for `t3_iso_scale_mass_8x`, per-group `L_raw` depth counting for 3-layer BiGANs.
   - `law4-energy-v10 -> v11`: restored generator-free 2D Critic NS specifically to fix the `0.013` miss on `t3_iso_scale_mass_8x` (`0.3375 vs 0.35`) and `t3_iso_horizon_saddle_shift` (`0.9062 vs 0.92`).
   - `law4-energy-v11 -> unified-v12`: added (i) the prime-67 CountSketch period-2 flip-bifurcation controller (`ema_fg < 0.60`, `s_therm >= 0.5`, `flip_thresh = 0.45`, `0.04 * flip_excess`) specifically to crack `grid100_res16_tanh`, and (ii) the conditional-fiber $K=2$ triangle-inequality radius multiplier (`min(2.0, (||g0||+||g1||)/||g0+g1||)`, `p_dec = 0.9995`) specifically to crack `model_ae_ddgan_gaussian_chain` and `model_ae_ddgan_bimodal_chain`!
2. **Codebase Complexity Explosion Confirms Mechanism Accumulation:** While claiming to distill "Four Laws", direct implementation size grew monotonically from `4,313` LOC (`477` branches) in `unified-v10`, to `4,713` LOC in original `unified-v11` (`8caab7bde`), to `7,076` LOC (`816` branches) in current `unified-v11`, to **`8,564` direct LOC (`1,057` AST branches: `818` `if` + `239` `IfExp`)** in `unified-v12` — with specialized guards conditioned on exact task signatures (`dim == 2 and d_z == 4`, `is_cond_fiber and self._protected.shape[0] == 2`, `generator_free`, `ema_fg < 0.60 and s_therm >= 0.5`).
3. **Single-Seed (Seed `0`) Deterministic Coupling:** With fixed `seed = 0`, `deterministic_orthogonal` initialization, and fixed batch streams, the entire trajectory of all `48` `discriminator_stability` tasks is a deterministic function of code constants. Tuning constants (`3.52`, `1.30`, `0.60`, `0.45`, `0.9995`, `0.0061`, `2.9`, `1.5`, `0.014`) until `46/48` pass on seed `0` provides **zero statistical estimate of cross-seed pass probability** and creates razor-thin boundary passes (e.g., `grid100_res16_tanh` flipping from `PASS` to `FAIL` under a **1-ULP** `float64` scalar rounding change in PR #615!).
4. **Unequal Tuning Budget Across Comparators:** As recorded in [`reports/forge/technique-inventory.md`](../forge/technique-inventory.md#L51-L62) and Section 1.1, `unified` + `law4-energy` + Round 1–4 tracks consumed **`78.27` GPU-h across `190` readouts**, whereas `empf` consumed `8` cards (`8.40` GPU-h), `rhbi` `12` cards (`10.12` GPU-h), `minimal` `2` cards, and `e22` `1` card (untuned). Furthermore, the `docs/four-laws.md` prose on line 211 discloses iterative tuning only for `unified_v11` (`unified-v1..v11`), **omitting** `law4-energy-v1..v11`, `unified-v12`, and the Round 1–3 feeder tracks (`t1-*`, `t2-*`, `rgdt-*`, `t4-*`, `cond-fiber`).

---

## 2. Honest Disclosure, Calibration Status (`BLOCKED`), and Governance Drift Audit

### 2.1 What `docs/four-laws.md` Discloses Honestly
- **Calibration `BLOCKED`:** [`docs/four-laws.md`](../../docs/four-laws.md#L3) and [line 210](../../docs/four-laws.md#L210) accurately state that screening calibration remains **`BLOCKED`** under frozen [`configs/forge/calibration/criteria-v1.json`](../../configs/forge/calibration/criteria-v1.json) (`sha256: 6eac2531...`), verified in [`reports/forge/calibration/r3-calibration-holdout.json`](../forge/calibration/r3-calibration-holdout.json#L2) (`"adoption": "BLOCKED"`, `"reference_positives_observed": 0`, rev-19 Tier-1 smoke wall `1,229.0s > 900.0s`, smoke/reference ratio `0.7009 > 0.10`, and 3-task cheap screen false-accept infeasibility across partial solvers).
- **`default_adoption: false`:** Verified verbatim in [`configs/forge/selections/family-current-v1.json`](../../configs/forge/selections/family-current-v1.json#L2) (`"default_adoption": false`).
- **Open Confirmatory Gaps:** Line 210 explicitly acknowledges that *"multi-seed confirmatory promotion and untouched real-data image FID confirmation remain open."*
- **12 `BLOCKED` Tier-4 Stretch Hosts:** Line 220 accurately distinguishes the `12` unimplemented `STRETCH_VECTOR_HOSTS` tasks (`BLOCKED` at preflight) from numerical `FAIL`.

### 2.2 Governance Drift & Nondisclosure Gaps Found by Audit
1. **Default Promotion Despite `BLOCKED` Calibration:** [`r3-calibration-holdout.md`](../forge/studies/r3-calibration-holdout.md#L200) and [`EXPERIMENTATION.md`](../../EXPERIMENTATION.md#L92-L99) explicitly state that default promotion requires accepted calibration and preregistered multi-seed robustness, and instructed *"Keep `DEFAULT_RECIPE == 'empf'` until calibration passes."* Nevertheless, `particlegan/recipes.py:67` promoted `DEFAULT_RECIPE` to `"unified_v11"` (v0.11.0) and then `"unified_v12"` (v0.12.0) by owner override while Forge's own selection pin [`family-current-v1.json`](../../configs/forge/selections/family-current-v1.json#L1063-L1081) still pins `"unified-v11"` and `"default_adoption": false`.
2. **Stale Failure List in `docs/four-laws.md` (Lines 212–219):** The "Honest Failure" section lists only the 7 historical failures of `unified-v11` (copied from `r4-solution-complete.md`) and **omits**:
   - `unified-v12`'s genuine numerical failure on `interp_img_patch_shift` (`interp_peak = 0.4915 < 0.78`, `finite_fraction = 1.0`, `endpoint_rmse = 0.0967`, `interp_rmse = 0.1002` in [`readout-b51ab67c`](../forge/records/readout-b51ab67c4d97526186e23651.json) — falsely blamed on the `0.99999994` float32 scorer bug in [`law4-energy-champion.md`](../forge/studies/law4-energy-champion.md#L131)!).
   - `unified-v12`'s `0/2` FAIL on the post-freeze unseen ByteGAN benchmarks ([`reports/forge/bytegan-stretch/summary.json`](../forge/bytegan-stretch/summary.json)).
   - `unified-v12`'s 5 numerical failures, 2 timeouts, and 2 runner errors in the Colab NVIDIA L4 transfer waves ([`reports/forge/technique-receipts/`](../forge/technique-receipts/)).
3. **Uncommitted Holdout Receipts:** The `18/18` untouched Tier-3 holdout claim and `5/5` shift-recovery claim for `unified-v12` have **zero committed `readout-*.json` files** in `reports/forge/records/` (detailed further in [P9 Audit](p9-receipts-efficiency-audit.md)). Moreover, `r3-tier3-long-horizon-holdout-v1` is **no longer genuinely untouched** by Phase 4: programmatically intersecting its `18` tasks with `discriminator_stability-v19 | t4-strange-architectures-v1 | r4-occam-ablation-v1` shows **`13/18` tasks directly overlap** (`ae_gan_hold`, `five_word_joint_hold`, `five_word_joint_smoke`, `grid100`, `mode_hold`, `ring_extension`, `ring_hold`, `rotated100`, `staggered100`, `two_pole`, `unused_token_hold`, `zero_tanh_rail_escape`, `zero_tanh_smoke`), and including `t3-shift-recovery-diagnostic-v1` (`target_shift_recovery`) brings total overlap to **`14/18`** — leaving only **`4/18`** (`clockfree_audit`, a legacy alias of `clockfree_audit_measurement_v1`, plus the 3 `*_14k` continuations `grid100_14k`, `rotated100_14k`, `staggered100_14k` of already-tuned 7k lattice tasks).

---

## 3. Exhaustive Autopsy of Failures, Near-Misses, Regressions & Unseen Generalization

### 3.1 Evolution of Failures Across `unified-v10`, `unified-v11`, and `unified-v12` (RTX A6000, Seed `0`)

| Task | `unified-v10` (`readout-6da9dee1`) | `unified-v11` (`8caab7bde`, `fc5b67a7` / `b96d9de4`) | `unified-v11` (`develop 87d72c39b`, via `v6` `2b91f35b`) | `unified-v12` (`ff41cf173`, `b51ab67c` / `c1ee5c3b`) | `unified-v12` Combined (`+PR #615`) / L4 Transfer | Root-Cause Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `staggered100` | **PASS** (`ctr=0.146`) | **FAIL** (`final ctr=0.1649`, 100k holdout PASS) | — | **PASS** (`b51ab67c`, `prec=0.9909`, `ctr=0.0618`) | **PASS** (A6000 Phase-4 `ctr=0.1243`) | Regressed `v10 -> v11` due to critic NS equilibrium jitter; recovered by 0-NS Frob-MP critic in `law4-energy-v10`/`v12`. |
| `grid100_res16_tanh` | **FAIL** (live suf) | **FAIL** (100k holdout PASS `0.9884`, missed live suf) | **FAIL** | **PASS** (`c1ee5c3b`: `prec=0.9907`, `ctr=0.1729`) | **FAIL** combined A6000 (step-6250 `modes=99` 1-ULP dip, uncommitted) & **FAIL** on NVIDIA L4 (`dadc826a`, `ctr=0.1214`, `accuracy_pass=True`, live suf miss) | **Razor-edge chaotic sensitivity:** prime-67 CountSketch flip controller sits right at the stability margin; a 1-ULP scalar change or GPU arch change (L4) drops 1 mode at step 6250. |
| `five_word_joint_hold` | **PASS** (`tv=0.0258`) | **FAIL** (`23/25` checks, dips at 3001 & 3334) | — | **PASS** (`b51ab67c`, `tv=0.0209`) | **PASS** (A6000 & L4) | Regressed `v10 -> v11`; recovered by 0-NS critic equilibrium damping. |
| `model_ddgan_bimodal_chain` | **FAIL** | **FAIL** (`8/24` checks, `t1_q0 eig_max=3.1085`) | — | **PASS** (`b51ab67c`, `J1=8,015 J`) | **PASS** | Recovered in `law4-energy-v10`/`v12` via lower critic equilibrium agitation. |
| `model_ae_ddgan_gaussian_chain` | **FAIL** | **FAIL** (`0/24` checks, `t1_q0_mean=0.7536`) | — | **PASS** (`b51ab67c`, `nmse=0.0448`, `t1_q0_cdf=0.2098`) | **INCOMPLETE (2× timeout `180.3s`) on NVIDIA L4** (`db86e31f`, `0069d955`) | Cracked on A6000 by bespoke $K=2$ triangle-inequality radius compensation (`_compute_two_obj_pyth_scales`), but exceeds wall budget on L4. |
| `interp_img_patch_shift` | **PASS** (`endpoint_hq=1.0`, `interp_hq=1.0`) | **FAIL** (`finite_fraction=0.99999994 < 1.0`, `endpoint_hq=0.0`) | — | **FAIL** (`b51ab67c`: **`finite_fraction=1.0`**, `interp_peak=0.4915 < 0.78`, `endpoint_rmse=0.0967`, `interp_rmse=0.1002`) | **FAIL** on L4 (`a7b3e4a2`: `finite_fraction=0.99999994`) | **Major regression & misattribution:** `unified-v10` PASSED (`1.0/1.0`); `unified-v12` on A6000 genuinely fails with `0.49`-peak ghost double-exposures (`law4-energy-champion.md` falsely blamed the float32 scorer edge!). |
| `interp_saddle_barycentric` | **FAIL** (`bhq=0.7183`) | **FAIL** (`bhq=0.6795`, `mae=0.0741`) | — | **FAIL** (`b51ab67c`: `bhq=0.7027 < 0.78`, `mae=0.0679`) | **FAIL** on L4 (`3ed32662`: `bhq=0.6522`) | Persistent manifold-barycenter interpolation shortfall across all `unified-v*` (whereas `rhbi-v12` PASSES at `bhq=0.8358`, `readout-47591bd9`). |
| `grid100_mlp16` & 4 others on `develop 87d72c39b` | **PASS** | **PASS** (`8caab7bde`) | **FAIL** (`11/16` Occam: regressed `mlp16`, `1d_stab`, `ddgan`, `8x`, `unequal_width`) | **PASS** (`c1ee5c3b`: `prec=0.9914, ctr=0.0727`) | **PASS** | Demonstrates silent regression fragility when editing `_unified_v11/` (`78ca41002..cb5437de9`). |
| **12 Tier-4 Stretch Hosts** | `BLOCKED` | `BLOCKED` (`add854c6`) | — | `BLOCKED` (unimplemented adapters) | `BLOCKED` | `misgan_*`, `pacgan_*`, `alice_spiral_bigan`, `infogan_*`, `cgan_*`, `cramer_*`, `cyclegan_*`, `ring16_bigan_continuous` remain unmeasured. |

### 3.2 Smoking-Gun Evidence of Overfitting: Post-Freeze Unseen Tasks & Cross-Architecture/GPU Transfer

Two natural experiments already committed on `origin/develop` **after** `unified-v12` was frozen (`ff41cf173`) directly falsify claims of universal law-complete generalization:

1. **Post-Freeze Unseen Discrete High-D Distribution — `ByteGAN` (PR #632, [`reports/forge/bytegan-stretch/summary.json`](../forge/bytegan-stretch/summary.json)):**
   - Two brand-new tasks on the 8D discrete hypercube $`\{-1,+1\}^8`$ (`256` uniform `uint8` modes, `z_dim = 8`, zero-sigma `ParticlePrior`): `bytegan_hypercube8` and `bytegan_bigan_dictionary`.
   - **Result:** **`unified-v12` scored `0 / 2` FAIL** (as did all 4 other families: `rhbi-v12`, `bispf`, `empf-v8`, `bcap-pure`):
     - `bytegan_hypercube8`: only **`139 / 256` valid unique bytes** (`154` raw unique, `rms_bit_error = 0.0875`) — **45.7% of discrete modes missing/collided!**
     - `bytegan_bigan_dictionary`: **`170 / 256` valid unique bytes** (`67` collisions, `86` orphans, `rms_bit_error = 0.1505 > 0.12`), **`0 / 256` `recon_valid_exact_bytes`** (`recon_rms_bit_error = 0.8635`), and `154 / 256` `encoder_particle_bijection_bytes`. Notably, much older `bcap-pure` outperformed `unified-v12` on `valid_unique_bytes` (`175` vs `170`) and `encoder_particle_bijection_bytes` (`183` vs `154`).
   - **Scientific Implication:** `unified-v12`'s Law-1 prototype atlas (`t1_max_prototypes = 256`) and Law-4 optimizer were heavily tuned on 2D lattices (`100` modes) and 5-word discrete sequences (`5` modes), yet fail catastrophically on an 8D `256`-mode uniform hypercube bijection — directly exposing non-generalization outside the seen task manifold.

2. **Cross-GPU & Architecture-Transfer Failures on NVIDIA L4 (`reports/forge/technique-receipts/`, Waves 7a/7b):**
   - Out of `76` `unified-v12` NVIDIA L4 task results (`67` PASS, `5` FAIL, `4` INCOMPLETE):
     - **`rotated100_affine_square_named_v1` — FAIL** (`f1786f5360e3436aaf063d34908f1f3d.json`, `abs_cov_trace_bias = 0.1069 > 0.10`, `accuracy_pass = False`): simply replacing the MLP generator with the identity-affine generator + Fourier-3 critic + uniform-square init breaks `rotated100`!
     - **`vector_unequal_width_published` — FAIL** (`24212e905bad46c7bd912e9a362605a8.json`, `component_covariance_error = 1.9237`): simply swapping the critic to the published `BatchDistanceDiscriminator` profile breaks `vector_unequal_width`!
     - **`grid100_res16_tanh` — FAIL** (`dadc826a15284be4a0f740290c4ca79b.json`, `accuracy_pass = True`, `ctr = 0.1214`, live-suffix miss): hardware floating-point reduction differences between RTX A6000 (Ampere) and NVIDIA L4 (Ada Lovelace) destroy the `7/7` standalone `t4-arch` claim!
     - **`model_ae_ddgan_gaussian_chain` — 2× `INCOMPLETE` (timeout at `180.3s`)** (`db86e31f...`, `0069d955...`), and **`grid100_affine_paired_laws_v1` / `grid100_release07_cloud_named_v1` — 2× `INCOMPLETE` (runner exit 1)**.

---

## 4. Why Passing an Operational Forge Gate != Proving a Mathematical Law

Even if a single global recipe passed `48/48` on `discriminator_stability`, `18/18` on Tier 3, and `7/7` on `t4-arch`, **passing operational Forge gates would still not constitute a proof of Laws 1–4**:

1. **Existential Witness vs. Universal Quantifier ($\exists \neq \forall$):** A mathematical law asserts a universal quantifier over distributions, architectures, initializations, and scales ($\forall P_Y \in \mathcal{P}, \forall (G, D) \in \mathcal{A}$). A Forge pass verifies only an existential witness on one finite list of synthetic tasks, one initializer (`deterministic_orthogonal`), one seed (`0`), and one GPU family (`RTX A6000`).
2. **Affirming the Consequent (Confounded Credit Assignment):** When a `8,564`-LOC recipe with `1,057` branches passes a gate after adding mechanism $M$, passing proves at most that $`(R \cup \{M\})`$ passes on seed `0` — not that $M$'s claimed theoretical mechanism caused the improvement rather than an incidental step-size damping, random trajectory perturbation, or interaction with another branch. (Indeed, Round-4 Arm 8 showed that `zero_tanh_rail_escape` still passed without the `fwAD` saturation probe `w_sat_pt` that had been credited for it!)
3. **Threshold-Cliff Aliasing:** Binary pass/fail gates hide proximity to catastrophic failure. `grid100_res16_tanh` passing standalone (`c1ee5c3b`) and failing by 1 mode at step 6250 after a 1-ULP scalar refactor (PR #615) or on an L4 GPU (`dadc826a`) shows that a binary `PASS` can sit with near-zero robustness margin.
4. **Proxy-Objective Divergence (`dcgan_performance_r256`):** `dcgan_performance_r256` trains for `13` updates on i.i.d. `Uniform[-1, 1]` random pixel noise with `particle_cloud` (`sigma=0.0`) and checks only finiteness and non-zero parameter motion. It provides **zero evidence** of high-resolution image convergence, FID, or energy-to-quality on natural images.
5. **Coordinate & Dimension Non-Invariance:** Several mechanisms in `unified-v12` explicitly branch on coordinate dimensions (`d == 1`, `dim == 2 and d_z == 2`, `dim == 2 and d_z == 4`, `d <= 4`, `5 <= d <= 8`, `d > 8`). A true geometric law must hold under orthogonal rotations, coordinate permutations, and affine rescalings without dimension-specific lookup tables.

---

## 5. Frozen Unseen Scientific Qualification Protocol (`FUSQP-v1`)

To separate genuine scientific generalization from Forge benchmark overfitting — while **strictly obeying [`AGENTS.md`](../../AGENTS.md)** (no seed-only tuning/search to improve a score; isolated named RNG streams; fixed task-owned architectures/priors; single unchanged global recipe; faithful numerical `FAIL` distinct from `BLOCKED`/`INCOMPLETE`) — we specify **`FUSQP-v1`**, an independently constructed, cryptographically frozen post-selection qualification suite.

### 5.1 Strict Two-Stage Separation (Screening vs. Confirmatory Freeze)

| Stage | Allowed Activity | Seed Policy | Governance Rule |
| :--- | :--- | :--- | :--- |
| **Stage 1: Development & Screening (Existing Forge)** | Architecture/theory research, ablations, recipe iteration | **Fixed seed `0` only** (`AGENTS.md`: zero seed-only optimization) | Results are strictly *hypothesis generation / screening*; never grant scientific superiority or `default_adoption: true`. |
| **Stage 2: Cryptographic Freeze & `FUSQP-v1` Confirmation** | **Zero code/hyperparameter edits** after registering `(recipe_sha256, source_commit, git_tree_sha256)` in a preregistered manifest | **Fixed disjoint confirmatory seed panel** $`S_{\text{conf}} = \{1001, 1002, \dots, 1010\}`$ ($K = 10$ seeds for fast vector/2D tasks; $K = 5$ seeds $`\{1001..1005\}`$ for 16-layer/image/long-horizon tasks), executed **once per frozen candidate** | Any post-hoc tweak invalidates the confirmation cohort and requires a brand-new candidate ID and a *fresh* unseen task partition (`FUSQP-v2`). **Never select among seeds; report the full distribution over all $K$ seeds.** |

### 5.2 Pillar A — Held-Out Architectures (Zero Overlap With Existing Forge Tasks)

Every task uses standard public `particlegan` API (`GANTrainer` / `E22Policy`), fixed `deterministic_orthogonal` init (plus one explicit separate Kaiming/Xavier init transfer cohort), and NO architecture-specific recipe overrides:

1. **`arch_swiglu_mlp8_mix32`:** 8-layer `SwiGLU` (`SiLU(W_1 x) ⊙ W_2 x`) generator + `GELU` LayerNorm critic on a 32-mode irregular 2D mixture (tests non-ReLU gated MLPs without LeakyReLU assumptions).
2. **`arch_resnet_groupnorm_img16`:** 12-block Residual Conv2d generator/critic with **`GroupNorm(8)`** (eliminating BatchNorm batch-coupling!) on 16×16 8-mode geometric shape patterns (tests Law 2 & Law 4.3 without BatchNorm smoothing).
3. **`arch_patch_vit_critic_img32`:** Lightweight 4-layer Patch-4 **Vision Transformer critic** (`LayerNorm` + Multi-Head Self-Attention + GELU MLP, `d_model=128`, 4 heads) paired with a ConvTranspose generator on 32×32 multi-color synthetic sprites (tests whether Law 2 `fwAD`/secant and Law 4 0-NS critic generalize to attention blocks!).
4. **`arch_aspect_ratio_sweep` (3 sub-tasks):** Extreme width/depth ratios holding parameter count $\approx 150\text{k}$ fixed on a 25-mode target:
   - *Wide-shallow:* `2 × 512` (tests wide Marchenko–Pastur scaling),
   - *Deep-narrow skip-free:* `24 × 64` (deeper than `grid100_mlp16`!),
   - *Hourglass bottleneck:* `256 -> 16 -> 256 -> 16 -> 256` (severe rank bottleneck in intermediate layers).

### 5.3 Pillar B — Mathematical-Invariant Transformations of Canonical Tasks

Tests whether "Laws 1–4" are genuinely invariant under symmetry transformations that preserve the underlying mathematical problem:

1. **`inv_ortho_rotation_d8`:** Embed a 2D 100-mode irregular mixture and a 4-mode unequal-width mixture into $\mathbb{R}^8$ via a fixed Haar-random orthogonal matrix $Q \in O(8)$ (plus zero-variance or tiny-noise orthogonal complement). **Defeats `dim == 2` special-casing!**
2. **`inv_affine_scale_sweep`:** Rescale target coordinates $Y \mapsto \alpha Y + b$ across **6 orders of magnitude** $`\alpha \in \{10^{-3}, 10^{-1}, 10^{+1}, 10^{+3}\}`$ with non-zero offset $b = (50\alpha, -30\alpha)$ and anisotropic diagonal scaling $\mathrm{diag}(10^{-2}, 10^{+2})$. **Directly tests Law 3 scale-invariance and Law 4.2 degree-0 gauge invariance!**
3. **`inv_latent_dim_sweep`:** Run canonical 100-mode and 16-mode tasks with $`d_z \in \{3, 5, 7, 16, 32\}`$ — explicitly excluding special-cased $`d_z \in \{2, 4\}`$ branches (`Sigma_push2` and Radon–Hurwitz quaternion)!
4. **`inv_batch_size_mismatch`:** Fix $M = 100$ modes while sweeping minibatch size $`B \in \{16, 32, 64, 256\}`$ (testing Law 1's $B \ll M$ Poisson count-variance regime down to $B/M = 0.16$).

### 5.4 Pillar C — Independently Generated Pathological Distributions

Generated by a separate deterministic task generator (`seed_data = 20261124`) completely disjoint from Forge's 140 task cards:

1. **`dist_poisson_multiscale_64`:** 64-mode **non-lattice** Poisson-disc mixture in $\mathbb{R}^2$ with log-uniform mode widths $\sigma_m \in [0.02, 0.40]$ ($20\times$ width ratio, vs $8\times$ in `t3_iso`) and Zipf mode weights $w_m \propto m^{-0.8}$ ($27.9\times$ mass ratio, $64^{0.8} = 27.86$).
2. **`dist_nested_rings_spirals_3d`:** Three concentric nested 1D rings + two intertwined 3D logarithmic spirals in $\mathbb{R}^3$ (nonzero Betti numbers $\beta_0 = 5, \beta_1 = 3$; non-convex, curved 1D manifolds in 3D, breaking 2D-only manifold heuristics).
3. **`dist_heavy_tail_student_t`:** 8-mode multivariate Student-$`t`$ mixture ($\nu = 2.5$, finite mean/variance, heavy polynomial tails — no Gaussian sub-exponential tail assumption!).
4. **`dist_hybrid_manifold_atoms`:** Continuous 2D Swiss-roll sheet (60% mass) coexisting with 8 isolated tight point-like Gaussian atoms (40% mass) in $\mathbb{R}^3$ (tests whether prototype coalescence destroys continuous sheets or blurs discrete atoms).
5. **`dist_continuous_drift_oscillation`:** Non-stationary 12-mode ring whose radius breathes continuously $R(t) = 2.0 + 0.8\sin(2\pi t / 2000)$ and rotates at constant angular velocity $\omega = 2\pi/4000$ (tests clock-free continuous tracking rather than one-shot step-function shifts!).
6. **`dist_discrete_simplex_d12`:** 128 randomly chosen vertices of the 12D hypercube $`\{-1, +1\}^{12}`$ (`z_dim = 12`, generalizing the unseen `ByteGAN` failure mode).
7. **`dist_real_image_fid_anchor`:** Standardized small-scale real-image benchmark (e.g., CIFAR-10 `32×32` or FFHQ-64 subset under fixed 50k clean FID + Precision/Recall + NVML total Joules-to-FID-threshold), closing the gap left open by mock-noise `dcgan_performance_r256`.

### 5.5 Pillar D — Distribution-Level Reporting & Qualification Criteria

Rather than binary single-seed PASS/FAIL tallies, every frozen candidate evaluated on `FUSQP-v1` reports a **standardized multi-metric qualification card**:

1. **Multi-Seed Pass Rate & Exact Uncertainty:** For each task, report `k / K` sustained passes with **exact Clopper–Pearson 95% confidence interval**; aggregate pass rate across each Pillar (A, B, C) separately (never hiding Pillar failures inside a pooled score).
2. **Distribution-Level Effect Sizes & Worst-Case Tail:** Across the $K$ confirmatory seeds, report median, IQR, **worst-case (`max` / 90th-percentile)** error metrics (`mass_tv`, `W1`/`Sliced-W2`, `center_rms_sigma`, `cov_frob_rms`, `mode_recall`, `precision`), and paired Cliff's $\delta$ / bootstrap 95% CI vs the frozen incumbent baseline.
3. **Failure Frequency & Failure Mode Taxonomy:** Separate `PASS`, numerical `FAIL`, `NONFINITE/NaN`, `TIMEOUT/INCOMPLETE`, `ERROR`, and `BLOCKED` explicitly; record catastrophic-failure rate (`NaN` or `< 50%` mode recall).
4. **True Energy-to-Quality (`Joules-to-Threshold`) & Sample Efficiency:** Measure **integrated hardware NVML energy** (`nvmlDeviceGetTotalEnergyConsumption` sampled at every evaluation checkpoint, NOT `step * avg_J_per_step`!) to reach:
   - First-pass quality threshold (`J_first`, `steps_first`),
   - Sustained 5-check stability (`J_sustained`, `steps_sustained`),
   - **Dynamic energy** (`J_dyn`) alongside total board energy (`J_tot`) and verified idle baseline power (`idle_W`, requiring `gpu_util == 0%` before and after, rejecting contended runs automatically).
   - Report `∞` (right-censored) for non-passing seeds and compare via survival / Kaplan–Meier median `J-to-PASS`.
5. **Endurance & Shift Recovery Dynamics:** Report late-horizon drift slope (`metric(14k) - metric(7k)`), minimum suffix quality over 20% terminal window, and shift-recovery half-life ($`t_{1/2}`$ updates to recover 50% and 95% of pre-shift quality).
6. **Qualification & Default-Adoption Gate:** A candidate earns **`default_adoption: true`** ONLY if: (i) Forge screening calibration passes an updated non-degenerate criteria profile, (ii) on `FUSQP-v1`, it achieves **zero catastrophic/NaN failures**, statistically significant non-inferiority on every Pillar (lower bound of paired bootstrap 95% CI $\ge 0$ on Pillar pass rates and median `J-to-PASS`), and $`\ge 80\%`$ multi-seed pass rate on Pillars A–C with strict source/hash lock.

---

## 6. Claim-by-Claim Audit Ledger (10-Category Taxonomy & 8-Point Rubric)

### Claim P8-1: *"One unchanged global recipe + seed 0 prevents overfitting across Forge's 48-task screening suite."*
- **Classification:** `Falsified claim`
- **1. Formulation:** Requiring a single global recipe configuration across `discriminator_stability` and banning seed-only optimization prevents research-level overfitting to the Forge benchmark suite.
- **2. Assumptions:** Assumes the fixed 48-task seed-0 suite is representative of unseen distributions, architectures, coordinate scalings, and hardware cohorts, and that human/agent iteration across 68 sequential direct-ancestry cards does not adaptively overfit the benchmark.
- **3. Supporting evidence:** Enforcing one global recipe genuinely prevents per-task hyperparameter tables (a real improvement over ad-hoc tuning) and `unified-v12` improves screening pass count to `46/48` (`29/29` required) on RTX A6000 seed `0`.
- **4. Counterexamples:** (a) Post-freeze unseen `ByteGAN` (`bytegan_hypercube8`, `bytegan_bigan_dictionary`): `unified-v12` fails `0/2` (`139/256` and `170/256` unique bytes, `0/256` exact reconstructions, [`bytegan-stretch/summary.json`](../forge/bytegan-stretch/summary.json)). (b) Cross-GPU/architecture transfer on NVIDIA L4 (`technique-receipts`): `unified-v12` fails `rotated100_affine_square_named_v1` (`f1786f53`), `vector_unequal_width_published` (`24212e90`), and `grid100_res16_tanh` (`dadc826a`). (c) 1-ULP step-6250 `modes=99` regression on `grid100_res16_tanh` when combined with PR #615.
- **5. Minimal falsification experiment:** Evaluate frozen `unified-v12` (`ff41cf173`) on any unseen task outside the training lineage (already executed in PR #632 ByteGAN and Waves 7a/7b L4 transfer) or across seeds `{1, 2, 3, 4, 5}` on `grid100_res16_tanh` and `staggered100`.
- **6. Prior research:** Dwork et al. (2015), *The Reusable Holdout: Preserving Validity in Adaptive Data Analysis* (Science); Recht et al. (2019), *Do ImageNet Classifiers Generalize to ImageNet?* (ICML); Gelman & Loken (2014), *The Statistical Crisis in Science*; Bouthillier et al. (2021), *Accounting for Variance in Machine Learning Benchmarks* (MLSys).
- **7. Computational verification:** Programmatically verified in [`scripts/four_laws_audit/p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) (`audit_p8_tuning_and_unseen`) and [`tests/four_laws_audit/test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py) (`test_p8_governance_calibration_and_unseen_failures`): 68 direct-lineage cards / 190 readouts / 78.27 GPU-h, `0/2` ByteGAN, `5` L4 FAILs + `4` L4 INCOMPLETE.
- **8. Confidence & uncertainty:** **Very High (`>95%`).** All failure receipts are directly verified in the repository; the only remaining uncertainty is the exact multi-seed pass-rate curve across the 46 currently passing seed-0 tasks (unmeasured).

### Claim P8-2: *"`unified-v12` passes `18/18` (`100%`) untouched Tier-3/14k holdouts and can be interpreted as a zero-tuning holdout validation."*
- **Classification:** `Falsified claim`
- **1. Formulation:** `unified-v12` achieved a verifiable `18/18` (`100%`) PASS on an *untouched* Tier-3 holdout (`r3-tier3-long-horizon-holdout-v1`, [`docs/four-laws.md`](../../docs/four-laws.md#L169), [line 193](../../docs/four-laws.md#L193)). This fails on two independent grounds: uncommitted provenance and direct holdout contamination.
- **2. Assumptions:** Assumes (a) immutable Forge readout receipts back the `18/18` score, and (b) the 18 tasks were genuinely untouched during `unified-v4..v12` and `law4-energy-v1..v11` selection.
- **3. Supporting evidence:** [`law4-phase2-energy-benchmark.json`](../forge/studies/law4-phase2-energy-benchmark.json) records `"tier3_holdout_pass": 18, "tier3_holdout_total": 18`, and [`law4-energy-champion.md`](../forge/studies/law4-energy-champion.md#L130) quotes plausible per-task metrics (`grid100_14k 0.9869`, `rotated100_14k 0.9825`, `staggered100_14k 0.9897`, `ring_extension 300/300`).
- **4. Counterexamples:** (a) **Zero committed `readout-*.json` receipts** exist in `reports/forge/records/` for `unified-v12` on `r3-tier3-long-horizon-holdout-v1` (or `t3-shift-recovery-diagnostic-v1`). (b) **Severe holdout contamination:** **`13/18` tasks** in `r3-tier3-long-horizon-holdout-v1` (`ae_gan_hold`, `five_word_joint_hold`, `five_word_joint_smoke`, `grid100`, `mode_hold`, `ring_extension`, `ring_hold`, `rotated100`, `staggered100`, `two_pole`, `unused_token_hold`, `zero_tanh_rail_escape`, `zero_tanh_smoke`) directly overlap the active screening views (`discriminator_stability-v19`, `t4-strange-architectures-v1`, `r4-occam-ablation-v1`), and including `t3-shift-recovery-diagnostic-v1` (`target_shift_recovery`) brings overlap to **`14/18`** — leaving only **`4/18`** (`clockfree_audit` [legacy alias of `clockfree_audit_measurement_v1`] and the 3 `*_14k` continuations of already-tuned 7k lattices), all of which were already inspected during `unified-v3..v11`!
- **5. Minimal falsification experiment:** Check `reports/forge/records/readout-*.json` for `candidate_id == "unified-v12"` and `goal == "r3-tier3-long-horizon-holdout-v1"` (returns empty set) and intersect `r3-tier3-long-horizon-holdout-v1` task IDs with active screening views (`13/18` core overlap, `14/18` including shift diagnostic).
- **6. Prior research:** Dwork et al. (2015) on holdout reuse degradation; Kaufman et al. (2012), *Leakage in Data Mining: Formulation, Detection, and Avoidance* (ACM TKDD).
- **7. Computational verification:** Verified in `test_committed_unified_v12_readouts_and_three_way_r256_discrepancy` (`"r3-tier3-long-horizon-holdout-v1" not in v12["committed_v12_goals"]`) and `test_non_progression_readout_metrics_and_tier3_holdout_overlap` (exact `13/18` / `14/18` / `4/18` sets).
- **8. Confidence & uncertainty:** **Very High (`100%` on provenance absence and task overlap).** The underlying `18/18` run likely occurred locally on `3bf87cea3+ff41cf173` before PR #622, but cannot be audited without committed cryptographic receipts and is not an untouched holdout.

### Claim P8-3: *"Screening calibration under frozen `criteria-v1.json` remains mathematically `BLOCKED`."*
- **Classification:** `Proven mathematical result`
- **1. Formulation:** Under frozen [`configs/forge/calibration/criteria-v1.json`](../../configs/forge/calibration/criteria-v1.json), neither `discriminator_stability` rev-19's 11-task Tier-1 smoke screen nor the cheap 3-task smoke screen satisfies the conjunction of cost, positive-reference, and false-accept criteria, so calibration acceptance is arithmetically `BLOCKED`.
- **2. Assumptions:** Frozen thresholds in `criteria-v1.json`: `minimum_reference_positives >= 1`, `maximum_false_accept_fraction <= 0.10`, `maximum_smoke_wall_seconds <= 900.0`, `maximum_smoke_to_reference_wall_ratio <= 0.10`.
- **3. Supporting evidence:** [`r3-calibration-holdout.json`](../forge/calibration/r3-calibration-holdout.json) proves: (i) `reference_positives_observed = 0` in the audited cohort, (ii) rev-19 Tier-1 smoke = `1,229.0s > 900.0s` and ratio `0.7009 > 0.10`, and (iii) on the 3-task screen (`two_pole`, `unused_token_hold`, `ae_gan_hold`), `5/5` partial solvers are `false_accept` (`FA/(FA+TR) = 5/(5+1) = 0.833 > 0.10`), requiring $\ge 45$ independent `true_reject` controls just to clear `0.10` if all 5 remain false accepts.
- **4. Counterexamples:** Even if `unified-v12` is entered into the 16-task `initial.json` calibration reference set as a positive (`16/16` PASS), conditions (ii) and (iii) STILL mathematically block calibration acceptance under frozen `criteria-v1.json`.
- **5. Minimal falsification experiment:** Re-evaluate the `criteria-v1.json` reducer adding `unified-v12` as a reference positive (`1 true_accept`) alongside the existing 5 partial solvers and 1 negative control: cost bounds on rev-19 still fail (`1229 > 900`, `0.70 > 0.10`) and 3-task false-accept rate remains `5/6 = 83.3% > 10%`.
- **6. Prior research:** Neyman–Pearson hypothesis testing & Clopper–Pearson exact binomial bounds (Clopper & Pearson, 1934); sequential screening protocol design.
- **7. Computational verification:** Verified in `test_p8_governance_calibration_and_unseen_failures` (`calibration_adoption == "BLOCKED"`, `default_adoption is False`).
- **8. Confidence & uncertainty:** **Very High (`100%` arithmetic certainty on frozen `criteria-v1.json`).**

### Claim P8-4: *"Promoting `DEFAULT_RECIPE = 'unified_v12'` while calibration is `BLOCKED` and `family-current-v1.json` sets `default_adoption: false` creates documented governance drift."*
- **Classification:** `Empirically supported causal mechanism`
- **1. Formulation:** Setting the public package default `DEFAULT_RECIPE = "unified_v12"` ([`particlegan/recipes.py`](../../particlegan/recipes.py#L67)) while Forge's calibration reducer outputs `"BLOCKED"`, [`family-current-v1.json`](../../configs/forge/selections/family-current-v1.json#L1063-L1081) still pins `"unified-v11"` with `"default_adoption": false`, and [`r3-calibration-holdout.md`](../forge/studies/r3-calibration-holdout.md#L200) requires keeping `DEFAULT_RECIPE == "empf"` until calibration passes causes divergence between published scientific qualification status and user-facing library defaults.
- **2. Assumptions:** Assumes the repository's published governance contracts ([`EXPERIMENTATION.md`](../../EXPERIMENTATION.md#L92-L99) and [`r3-calibration-holdout.md`](../forge/studies/r3-calibration-holdout.md#L200)) are intended to gate default promotion unless explicitly amended.
- **3. Supporting evidence:** Direct code and manifest inspection shows three mutually inconsistent states across the repository: `recipes.py` (`DEFAULT_RECIPE = "unified_v12"`), `family-current-v1.json` (`selected_candidate_id = "unified-v11"`, `default_adoption = false`), and `r3-calibration-holdout.json` (`adoption = "BLOCKED"`).
- **4. Counterexamples:** Repository maintainers are entitled to override defaults explicitly for pragmatic release engineering, *provided* `family-current-v1.json` and `docs/four-laws.md` explicitly distinguish an "uncalibrated engineering preview default" from a scientifically qualified default (currently left contradictory).
- **5. Minimal falsification experiment:** Inspect `particlegan.recipes.DEFAULT_RECIPE`, `configs/forge/selections/family-current-v1.json`, and `reports/forge/calibration/r3-calibration-holdout.json` simultaneously; alignment holds iff all three agree on candidate identity and adoption status.
- **6. Prior research:** Gebru et al. (2021), *Datasheets for Datasets*; Mitchell et al. (2019), *Model Cards for Model Reporting* (FAT* — separating benchmark qualification status from deployment defaults).
- **7. Computational verification:** Verified in `test_p8_governance_calibration_and_unseen_failures` (`DEFAULT_RECIPE == "unified_v12"`, `family-current-v1.json` pinned to `"unified-v11"` with `default_adoption is False`, `calibration_adoption == "BLOCKED"`).
- **8. Confidence & uncertainty:** **Very High (`100%` on observed manifest/code discrepancy).**

### Claim P8-5: *"Passing a Forge gate validates the corresponding Four-Law mathematical theory."*
- **Classification:** `Falsified claim`
- **1. Formulation:** Observing `PASS` on a Forge suite mapped to Laws 1–4 (`grid100`, `dcgan_performance_r256`, `t3_iso_*`, `t4_nash`, `t4_arch`) proves the mathematical statement of the corresponding law.
- **2. Assumptions:** Would require equivalence between a finite operational benchmark predicate (fixed seed `0`, fixed architectures, fixed coordinate scales, finite tolerance) and a universal mathematical theorem.
- **3. Supporting evidence:** Forge gates *do* provide falsification power (e.g., `r4-razor-no-chart-laguerre` collapsing `grid100` to `prec=0.2827` strongly refutes the hypothesis that Laguerre dual balancing is unnecessary for the tested architecture).
- **4. Counterexamples:** (a) `unified-v12` passes `46/48` on `discriminator_stability` yet fails `0/2` on unseen `ByteGAN` and fails architecture-transfer variants (`vector_unequal_width_published`, `rotated100_affine_square_named_v1`); (b) `zero_tanh_rail_escape` passed with `w_sat_pt` AND still passed without it (Arm 8, `mae=0.0026`), showing that passing the gate did not validate the `w_sat_pt` causal claim; (c) `dcgan_performance_r256` has no distribution-quality metric at all.
- **5. Minimal falsification experiment:** Apply a symmetry transformation that leaves the mathematical law invariant (e.g., rotate `grid100` into $\mathbb{R}^8$ or swap `BatchDistanceDiscriminator` on `vector_unequal_width`) and observe whether passing the original Forge gate guarantees passing the mathematically equivalent transform (already falsified on `vector_unequal_width_published` and `rotated100_affine_square_named_v1`!).
- **6. Prior research:** Popper (1959), *The Logic of Scientific Discovery* (asymmetry between falsification and verification); Lucic et al. (2018), *Are GANs Created Equal?* (NeurIPS); Henderson et al. (2018), *Deep Reinforcement Learning That Matters* (AAAI).
- **7. Computational verification:** Verified via `unified-v12` receipts (`b51ab67c` PASS on `vector_unequal_width` & `rotated100` vs L4 `24212e90` FAIL on `vector_unequal_width_published` & `f1786f53` FAIL on `rotated100_affine_square_named_v1`).
- **8. Confidence & uncertainty:** **Very High (`>95%`).** Standard deductive logic plus direct empirical counterexamples in the repo.

### Claim P8-6: *"Multi-seed post-freeze confirmatory evaluation (`FUSQP-v1`) is necessary to establish scientific superiority and safe public-default adoption without violating `AGENTS.md`."*
- **Classification:** `Necessary condition under stated assumptions`
- **1. Formulation:** Under the goal of establishing generalizable scientific superiority (rather than seed-0 benchmark leadership) while forbidding seed-only hyperparameter tuning (`AGENTS.md`), a two-stage protocol separating seed-0 screening from a strictly post-freeze, unseen, multi-seed confirmatory suite (`FUSQP-v1`) is necessary.
- **2. Assumptions:** Stochastic minibatch sampling, hardware floating-point non-associativity (Ampere vs Ada Lovelace), and initialization sensitivity induce non-zero run-to-run variance near stability boundaries; any single-seed observation has Clopper–Pearson 95% lower confidence bound of only `0.025` for $1/1$ PASS.
- **3. Supporting evidence:** Directly evidenced by (i) `grid100_res16_tanh` flipping from PASS (`c1ee5c3b`) to FAIL under a 1-ULP scalar change (PR #615) and on L4 (`dadc826a`), (ii) `r3-calibration-holdout.md` and `EXPERIMENTATION.md` already distinguishing seed-0 screening from post-freeze multi-seed promotion (`reports/forge/promotions/`), and (iii) the failure of `unified-v12` on post-freeze unseen `ByteGAN` and architecture-transfer hosts.
- **4. Counterexamples:** None for generalizable superiority; for pure software determinism/regression checks (`clockfree_audit`, bit-exact continuation), seed-0 testing remains sufficient.
- **5. Minimal falsification experiment:** Run frozen `unified-v12` (`ff41cf173`) across 5 preregistered unseen seeds `{1001..1005}` on `grid100_res16_tanh`, `staggered100`, `model_ae_ddgan_gaussian_chain`, and Pillar A/B tasks; if variance across seeds were identically zero and transfer were 100%, multi-seed unseen confirmation would be redundant (already contradicted by 1-ULP and L4 sensitivity).
- **6. Prior research:** Bouthillier et al. (2021), *Accounting for Variance in Machine Learning Benchmarks* (MLSys); Dror et al. (2019), *Deep Dominance — How to Properly Compare Deep Neural Models* (ACL); Agarwal et al. (2021), *Deep Reinforcement Learning at the Edge of the Statistical Precipice* (NeurIPS — IQM, stratified bootstrap CIs, performance profiles).
- **7. Computational verification:** Exact binomial lower bound: for $n=1, k=1$, $\text{Beta}^{-1}(0.025; 1, 1) = 0.025$ (95% CI `[0.025, 1.000]`); for $n=10, k=10$, 95% CI tightens to `[0.692, 1.000]`.
- **8. Confidence & uncertainty:** **Very High (`>95%`).** Standard statistical inference; implementation cost of `FUSQP-v1` is bounded (~2–4 GPU-hours per frozen release candidate).

### Claim P8-7: *"The `12` `BLOCKED` Tier-4 tasks represent missing engineering adapters rather than proved mathematical failure, yet leave multi-paradigm GAN generality (`MisGAN`, `PacGAN`, `BiGAN`, `CycleGAN`, `InfoGAN`, `cGAN`) empirically unverified."*
- **Classification:** `Unresolved question`
- **1. Formulation:** Because all 12 `STRETCH_VECTOR_HOSTS` tasks fail preflight with `"no registered public Forge adapter for this stretch host"`, `unified-v12`'s performance on incomplete-data imputation (`MisGAN` with and without `ppost`), packed critics (`PacGAN`), continuous BiGAN, InfoGAN, projection cGAN, CramerGAN, and unpaired CycleGAN is completely unmeasured.
- **2. Assumptions:** `AGENTS.md` strictly forbids reclassifying `BLOCKED` as `FAIL` or claiming generality over unmeasured paradigms.
- **3. Supporting evidence:** Verified in [`r3-calibration-holdout.md`](../forge/studies/r3-calibration-holdout.md#L140-L145) and [`readout-add854c6`](../forge/records/readout-add854c63ab9c700d22fc3ba.json).
- **4. Counterexamples:** Partial multi-objective/BiGAN evidence exists only on `five_word_joint_smoke/hold` (discrete 5-word BiGAN, PASS) and `bytegan_bigan_dictionary` (discrete 256-byte BiGAN, **FAIL `0/256` exact recon**).
- **5. Minimal falsification experiment:** Implement clean, non-assisting public Forge adapters for `ring16_bigan_continuous`, `misgan_block_neural_imputer`, and `cyclegan_unpaired_constellation` and run frozen `unified-v12` unmodified.
- **6. Prior research:** Li et al. (2019), *MisGAN* (ICLR); Lin et al. (2018), *PacGAN* (NeurIPS); Donahue et al. (2017) / Li et al. (2017), *BiGAN / ALICE*; Zhu et al. (2017), *CycleGAN*.
- **7. Computational verification:** Verified via `r3-calibration-holdout.md` and `technique-inventory.md` (`12 incompatible` in Tier 4 for `unified`).
- **8. Confidence & uncertainty:** **Very High (`100%` that these paradigms are currently unverified).**

---

## 7. Actionable Recommendations for `docs/four-laws.md` & Forge Governance

1. **Correct `docs/four-laws.md` Disclosures (without mutating historical receipts):**
   - Explicitly add `unified-v12`'s genuine failures (`interp_img_patch_shift` `interp_peak=0.4915` on A6000, `interp_saddle_barycentric`, unseen `ByteGAN` `0/2`, and L4 transfer failures) alongside `unified-v11`'s 7 historical failures.
   - Disclose the **full cumulative tuning budget** (`68` direct-ancestry cards / `190` readouts / `78.27` GPU-h across Round 1–4, `unified-v1..v12`, and `law4-energy-v1..v11`), not merely `unified-v1..v11`.
   - Remove the word *"untouched"* from `r3-tier3-long-horizon-holdout-v1` (`13/18` tasks directly overlap active screening views — `14/18` including `t3-shift-recovery-diagnostic-v1`, leaving only `clockfree_audit` and three `*_14k` continuations of tuned 7k lattices) and commit official Forge `readout-*.json` receipts before claiming `18/18` Tier-3 or `5/5` shift recovery for `unified-v12`.
2. **Resolve Governance Drift:** Either align `family-current-v1.json` and `DEFAULT_RECIPE` explicitly under a documented "uncalibrated research-preview default" policy (distinct from scientific qualification), or restore calibration gating before default changes.
3. **Implement `FUSQP-v1` Before Any Claim of Universal Law Validation:** Freeze `unified-v12` (or its Occam-pruned successor) and run Pillars A–D across the preregistered confirmatory seed panel $`S_{\text{conf}} = \{1001..1010\}`$ with zero post-freeze tuning.
