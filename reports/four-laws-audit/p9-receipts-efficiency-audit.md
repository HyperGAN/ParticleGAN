# P9 Audit: Primary Forge Receipt Verification, Efficiency Comparability, Energy-to-Quality, and Occam's Razor Minimality

**Audit Track:** P9 — Primary Receipt Integrity & Computational Efficiency / Occam Audit  
**Integration Branch:** `research/four-laws` (branched from `origin/develop` at `0d18245e9`)  
**Audited Documents & Codebase:** [`docs/four-laws.md`](../../docs/four-laws.md), [`reports/forge/studies/r4-occam-ablation-razor.md`](../forge/studies/r4-occam-ablation-razor.md), [`reports/forge/studies/law4-energy-champion.md`](../forge/studies/law4-energy-champion.md), [`reports/forge/studies/law4-optimizer-laws.md`](../forge/studies/law4-optimizer-laws.md), [`reports/forge/studies/r4-solution-complete.md`](../forge/studies/r4-solution-complete.md), [`reports/forge/studies/r4-minimal-core.md`](../forge/studies/r4-minimal-core.md), [`reports/forge/studies/law4-phase2-energy-benchmark.json`](../forge/studies/law4-phase2-energy-benchmark.json), [`reports/forge/records/readout-*.json`](../forge/records/), [`particlegan/_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py), and [`particlegan/_unified_v11/`](../../particlegan/_unified_v11/).  
**Reproducible Verification Script & Tests:** [`scripts/four_laws_audit/p9_verify_receipts.py`](../../scripts/four_laws_audit/p9_verify_receipts.py) and [`tests/four_laws_audit/test_p9_receipt_integrity.py`](../../tests/four_laws_audit/test_p9_receipt_integrity.py).

---

## Executive Verdict

1. **Three Broken File Links in `docs/four-laws.md` — Including Both Cited `unified-v12` Receipts:** Programmatic verification of all `77` local file links and `40` unique `readout-*` references in [`docs/four-laws.md`](../../docs/four-laws.md) proves that **`38/40` historical readouts exist and match their cited metrics**, whereas **both `unified-v12` receipts linked on line 169 (`readout-5002641a64ddaa91f1fc3b87.json`, `readout-731e752ff3c4f97e29a84145.json`) and the source link on line 175 (`particlegan/unified_v12.py`) DO NOT EXIST anywhere on `origin/develop` or in Git history.** (Two additional `unified-v12` hashes in [`law4-energy-champion.md`](../forge/studies/law4-energy-champion.md#L132-L133), `readout-7d00fc8c` and `readout-a85f705c`, also do not exist.) Instead, PR #620 (`ff41cf173`) committed **four different `unified-v12` readout files**: `readout-b51ab67c` (`46/48` `disc_stab`, `29/29` required), `readout-c1ee5c3b` (`7/7` `t4-arch`), `readout-624964f4` (`16/16` Occam), and `readout-59898e1a` (`6/6` `dcgan_performance`, **`41.39 ms/step`** — not `39.76` or `39.22 ms/step`). Zero committed `readout-*.json` receipts exist for `unified-v12`'s claimed **`18/18` Tier-3 endurance**, **`5/5` shift recovery**, **`6/7` combined `t4-arch`**, or **`39.76 ms/step` combined `r256`**.
2. **Three-Way `r256` Latency Discrepancy & Severe Comparability Confounds:** `docs/four-laws.md` mixes **three incompatible `r256` measurements** for `unified-v12` (`41.39 ms` committed Forge standalone vs `39.76 ms` uncommitted Forge combined vs `39.22 ms / 11.37 J/st` custom 50-step NVML profiler that reuses a single fixed batch pair!) and splices the uncommitted Forge `39.76 ms · 1243.9 MiB` with the profiler's `11.37 J/st` inside one table cell (line 193). Worse, the headline progression (`588.75 -> 130.13 -> 89.19 -> 39.76 ms`) is **NOT precision-matched**: `rhbi-v12` (`588.75 ms`, `3750.6 MiB`) ran in **`float32` NCHW**, whereas every newer champion ran in **`bf16_channels_last`**! Our controlled idle RTX A6000 experiment proves that switching `rhbi-v12` alone to `bf16_channels_last` cuts its latency by **`-27.5%`** (`319.05 -> 231.28 ms`), VRAM by **`-41.9%`** (`4253.7 -> 2472.3 MiB`), and energy by **`-26.6%`** (`87.24 -> 64.05 J/st`). Additional confounds include **shared-GPU contention** (`law4-energy-v10`/`v11` Forge readouts have `contended=True`; `minimal-v2`'s `55.50 ms` was shared-load vs `44.98 ms` idle on our GPU 0), **`68.15 W` vs `116.74 W` idle-power drift**, and a **dynamic-energy inversion** (`law4-energy-v10` actually uses *less* dynamic energy per step, `8.45 J/st`, than `unified-v12`'s `8.68 J/st`).
3. **Genuine Algorithmic Speedup Confirmed Locally — With Clear Bottleneck Shift:** Despite the reporting confounds, our strictly idle RTX A6000 micro-benchmark (`util_before=(0, 0)`) independently confirms `unified-v12`'s genuine `~1.81×` speedup over current `unified-v11` on `r256` (**`36.61 ms/step` / `11.20 J/st`** vs `66.38 ms/step` / `18.86 J/st`), driven by a **`16.2×` optimizer collapse** (`39.81 ms [51.0%] -> 2.45 ms [6.1%]`). With model forward/backward now taking `76.0%` (`30.72 ms`), the single largest remaining algorithmic bottleneck is `d_penalty` (`5.37 ms`, `13.3%`).
4. **Occam's Razor Was Reversed, Not Enforced, Between `unified-v10` and `unified-v12`:** Replacing `unified-v12`'s Core LOC with `—` (line 193) and keeping `unified-v11` at stale `4,713` LOC conceals a massive code explosion: `unified-v11` grew by **`+50.1%`** to **`7,076` LOC (`816` branches)**, and **`unified-v12` totals `8,564` direct LOC (`1,057` AST branches: `818` `if` + `239` `IfExp`)** plus transitive imports from 8 sibling packages — **`9.53×` larger than `minimal-v2` (`899` LOC)**. Proven Round-4 redundant epicycles (`d_z == 4` Radon–Hurwitz quaternion branch, `_fwad_generator_probe`/`w_sat_pt`, `rgdt_conformal_horizon`) remain present in `unified-v12`, and **6 of 7 major mechanisms added in `v11`/`v12` have zero single-factor destructive knockouts**.

---

## 1. Exhaustive Receipt & Link Integrity Audit (`docs/four-laws.md`)

Executed via `PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p9_verify_receipts.py`:

### 1.1 Broken Links & Missing `unified-v12` Receipts

| Location | Cited Target | Status on `origin/develop` (`0d18245e9`) | Actual Committed Replacement / Ground Truth |
| :--- | :--- | :---: | :--- |
| [`docs/four-laws.md:169`](../../docs/four-laws.md#L169) | `../reports/forge/records/readout-5002641a64ddaa91f1fc3b87.json` | **BROKEN (MISSING)** | [`readout-b51ab67c4d97526186e23651.json`](../forge/records/readout-b51ab67c4d97526186e23651.json) (`unified-v12-disc-stab-study`, PR #620 `ff41cf173`) |
| [`docs/four-laws.md:169`](../../docs/four-laws.md#L169) | `../reports/forge/records/readout-731e752ff3c4f97e29a84145.json` | **BROKEN (MISSING)** | [`readout-c1ee5c3bb7ac4f432532da63.json`](../forge/records/readout-c1ee5c3bb7ac4f432532da63.json) (`unified-v12-t4-arch-study`, PR #620 `ff41cf173`) |
| [`docs/four-laws.md:175`](../../docs/four-laws.md#L175) | `../particlegan/unified_v12.py` | **BROKEN (MISSING)** | [`particlegan/_unified_v12/optimizer.py`](../../particlegan/_unified_v12/optimizer.py) + [`particlegan/_unified_v11/`](../../particlegan/_unified_v11/) + [`particlegan/recipes.py:2816`](../../particlegan/recipes.py#L2816) |
| [`law4-energy-champion.md:132`](../forge/studies/law4-energy-champion.md#L132) | `readout-7d00fc8c` (`r4-occam-ablation-v1`) | **MISSING** | [`readout-624964f4e46793593bc147e4.json`](../forge/records/readout-624964f4e46793593bc147e4.json) (`unified-v12-occam-study`, `16/16` PASS) |
| [`law4-energy-champion.md:133`](../forge/studies/law4-energy-champion.md#L133) | `readout-a85f705c` (`dcgan_performance`, `39.74 ms`) | **MISSING** | [`readout-59898e1a77c4b179c9aebbc1.json`](../forge/records/readout-59898e1a77c4b179c9aebbc1.json) (`unified-v12-dcgan-perf-study`, **`41.39 ms/st`** median, `p95=42.05 ms`, `1243.4 MiB`) |

**Root Cause Analysis:** Neither `readout-5002641a...`, `readout-731e752f...`, `readout-7d00fc8c...`, nor `readout-a85f705c...` ever existed in any Git commit (`git log --all` verified empty). They were recorded as string fields inside [`reports/forge/studies/law4-phase2-energy-benchmark.json`](../forge/studies/law4-phase2-energy-benchmark.json) during an earlier pre-finalization local queue run (on combined tree `3bf87cea3+ff41cf173`), while PR #620 (`ff41cf173`) committed the four official standalone readouts (`b51ab67c`, `c1ee5c3b`, `624964f4`, `59898e1a`). When PR #622 (`63f15655b`) updated `docs/four-laws.md` and `law4-energy-champion.md`, it copied the uncommitted hashes from the JSON file instead of the committed filenames in `reports/forge/records/`.

### 1.2 Complete Audit of All 4 Committed `unified-v12` Readouts vs. `docs/four-laws.md` Claims

| Claim in `docs/four-laws.md` | Committed Receipt on `origin/develop` | Verbatim Receipt Facts (Hardware, Revision, Scope, Counts) | Audit Verdict |
| :--- | :--- | :--- | :---: |
| **`46/48` Tiers 1–2 `disc_stab` (`29/29` req = `100%`, incl. `2/2` AE-DDGAN, `3/3` precision, `3/3` Nash, `2/2` `t3_iso`)** | [`readout-b51ab67c4d97526186e23651.json`](../forge/records/readout-b51ab67c4d97526186e23651.json) | `cand_rev=f16b8465...`, `scope=current`, RTX A6000 GPUs 0/1, `contended=0/48`. Exact v19 split: `11/11` T1 req + `4/4` T1 stretch + `2/2` T1 diag + `18/18` T2 req + `7/7` T2 held + `4/6` T2 stretch = **`46/48` PASS**, **`29/29` required PASS**. Only 2 FAIL: `interp_img_patch_shift`, `interp_saddle_barycentric`. | **VERIFIED** *(after repairing broken link `5002641a -> b51ab67c`)* |
| **`7/7` standalone `t4-arch`** | [`readout-c1ee5c3bb7ac4f432532da63.json`](../forge/records/readout-c1ee5c3bb7ac4f432532da63.json) | `cand_rev=f16b8465...`, `scope=research_diagnostic`, RTX A6000 GPUs 0/1, `contended=0/7`. **`7/7` PASS** (`grid100_res16_tanh`: `prec=0.9907, ctr=0.1729, abs_bias=0.0805`; `grid100_mlp16`: `prec=0.9914, ctr=0.0727`). | **VERIFIED** *(after repairing broken link `731e752f -> c1ee5c3b`)* |
| **`16/16` Occam (`r4-occam-ablation-v1`)** | [`readout-624964f4e46793593bc147e4.json`](../forge/records/readout-624964f4e46793593bc147e4.json) | `cand_rev=f16b8465...`, `scope=research_diagnostic`, RTX A6000 GPUs 0/1, `contended=0/16`. **`16/16` PASS** (`12` fresh + `4` certified reused from `t4-arch`). | **VERIFIED** |
| **`r256`: `39.76 ms · 1243.9 MiB` (line 193 table) vs `39.22 ms / 11.37 J/st` (line 169 prose)** | [`readout-59898e1a77c4b179c9aebbc1.json`](../forge/records/readout-59898e1a77c4b179c9aebbc1.json) *(only committed Forge DCGAN readout)* + [`law4-phase2-energy-benchmark.json`](../forge/studies/law4-phase2-energy-benchmark.json) | **THREE-WAY DISCREPANCY:** (1) Committed Forge `readout-59898e1a` (standalone `ff41cf173`, `contended=False`, 3w+10m fresh batches): **`41.39 ms/st`** median (`p95=42.05 ms`, `1243.44 MiB`, `387.2 img/s`, `J_tot=500.42 J` over 13 steps); (2) Line-193 `39.76 ms · 1243.9 MiB`: **UNCOMMITTED** combined `3bf87cea3+ff41cf173` Forge run; (3) Line-169 `39.22 ms / 11.37 J/st`: custom **50-step NVML profiler** (`peak_allocated_mib=1219.93 MiB`, `idle_power_watts=68.15 W`, `dyn=8.68 J/st`, **reusing 1 fixed batch pair**). | **DISCREPANCY + SPLICED CELL** |
| **`18/18` Tier-3 endurance (`r3-tier3-long-horizon-holdout-v1`)** | *None in `reports/forge/records/`* | Only recorded as integer counters (`"tier3_holdout_pass": 18`) in `law4-phase2-energy-benchmark.json` and prose in `law4-energy-champion.md`. | **UNCOMMITTED / NO RECEIPT** |
| **`5/5` shift recovery (`t3-shift-recovery-diagnostic-v1`)** | *None in `reports/forge/records/`* | 4 of 5 tasks appear individually in `b51ab67c` / `c1ee5c3b`; `target_shift_recovery` has **no committed RTX A6000 `readout-*.json`** for `unified-v12`. | **UNCOMMITTED / NO RECEIPT** |
| **`6/7` combined `t4-arch` (`grid100_res16_tanh` 1-ULP dip)** | *None in `reports/forge/records/`* *(corroborated on L4 in `technique-receipts/dadc826a...`)* | No committed RTX A6000 combined `t4-arch` readout exists; NVIDIA L4 wave 7a independently confirms `grid100_res16_tanh` FAIL (`ctr=0.1214`, `accuracy_pass=True`, live-suffix miss). | **UNCOMMITTED ON A6000** |
| **`interp_img_patch_shift` attributed to `0.99999994` float32 scorer edge (`law4-energy-champion.md:131`)** | [`readout-b51ab67c4d97526186e23651.json`](../forge/records/readout-b51ab67c4d97526186e23651.json) | **MISREPRESENTATION:** Unlike `unified-v11` (`readout-fc5b67a7`, `finite_fraction=0.99999994 < 1.0`), `unified-v12` on A6000 had **`finite_fraction=1.0`** and **genuinely failed numerically** (`interp_peak=0.4915 < 0.78` double-exposure ghost, `endpoint_rmse=0.0967 > 0.09`, `interp_rmse=0.1002 > 0.10`). | **FALSIFIED ATTRIBUTION** |

### 1.3 Audit of All 38 Existing Readouts & Other Subtle Discrepancies in `docs/four-laws.md`

All `38` other readouts linked in `docs/four-laws.md` exist and match their cited metrics (`t2-secant-v2` `87.96 ms / 1308.46 MiB`; `t1-coupling-v5` `img_bars4` `4/4, hq=0.96875, tv=0.15625`; `t4-nash-v3` `39.68 ms / 1372.86 MiB`, `3/3` Nash; `t4-arch-v3` `7/7`, `grid100_mlp16 0.9814`, `res16_tanh 0.9856`, `res16_scaled_tanh 0.9868`, `zero_tanh_rail_escape 0.00105`; Round-4 Arms 1–8 knockouts; `unified-v11` `41/47`, `89.19 ms / 1372.60 MiB`, `6/7` `t4-arch`, `3/3` Nash, `5/5` shift, `2/2` `t3_iso`, `3/3` endurance), with **two minor denominator discrepancies** worth flagging:
- `t1-prior-v2` ([`readout-379720b9`](../forge/records/readout-379720b9859815defff0efaa.json)): receipt contains **`42` task results (`29 PASS, 12 FAIL, 1 INCOMPLETE`)**, whereas line 127 states `29/41` (omitting the 1 `INCOMPLETE` task from the denominator).
- `t1-coupling-v5` ([`readout-b87c22fb`](../forge/records/readout-b87c22fbdb500d43981e839b.json)): receipt contains **`44` task results (`31 PASS, 10 FAIL, 3 INCOMPLETE`)**, whereas line 128 states `31/41` (omitting the 3 `INCOMPLETE` tasks from the denominator).

---

## 2. Full `r256` Progression Table & Comparability Confound Audit

### 2.1 Verbatim Progression Table Cross-Check (All 14 Audited Candidates)

| Candidate | Committed Forge `dcgan_perf` Readout | `training_precision` | Device | Contended? | Forge `r256` Med (`ms`) · Alloc (`MiB`) | Claimed in `docs/four-laws.md` | Phase-2 Profiler `r256` (`ms` · `J/st` · `dyn J/st` · `idle W`) | Comparability Flags |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`rhbi-v12`** | `readout-9917dee7` | **`float32`** | `GPU 1` | `None` | `588.75` · `3750.6` | `588.75` · `3750.6` | — | **PRECISION CONFOUND (FP32 vs BF16 CL)** + stale pre-PR-#630 (`319.05 ms` today!) |
| **`empf-v8`** | `readout-9672bf97` | `bf16_channels_last` | `GPU 1` | `None` | `130.13` · `1435.4` | `130.13` · `1435.4` | — | Unverified contention (`cont=None`); other `empf-v8` receipts range `129.72–173.05 ms` |
| `t2-secant-v2` | `readout-14cc2a44` | `bf16_channels_last` | `GPU 1` | `None` | `87.96` · `1308.5` | `87.96` · `1308.46` | — | Verbatim match (`cont=None`) |
| `cond-fiber-v1` | `readout-8d2e547c` | `bf16_channels_last` | `GPU 0` | `None` | `87.50` · `1308.4` | `87.50` · `1308.4` | — | Verbatim match (`cont=None`) |
| `unified-v10` | `readout-90580c7a` | `bf16_channels_last` | `GPU 0` | `None` | `89.66` · `1372.6` | `89.66` · `1372.6` | — | Verbatim match (`cont=None`) |
| **`unified-v11`** | `readout-1e5fb235` | `bf16_channels_last` | `GPU 0` | `None` | `89.19` · `1372.6` | `89.19` · `1372.6` | `85.81 ms` · `23.70 J` · `15.98 J` · `89.7 W` | Historical `8caab7bde`; current `develop` post-#615 is `73.21 ms` (profiler) / `66.38 ms` (our idle GPU 0) |
| `r4-razor-no-ns-v2` | `readout-0219c6e2` | `bf16_channels_last` | `GPU 0` | `None` | `76.17` · `1372.6` | `76.17` · `1372.6` | — | Verbatim match |
| `r4-razor-autograd-critic` | `readout-4b04c4bf` | `bf16_channels_last` | `GPU 0` | `None` | `133.01` · `1850.3` | `133.01` · `1850.3` | — | Verbatim match |
| **`minimal-v2`** | `readout-d50a87c6` | `bf16_channels_last` | `GPU 1` | `None` *(shared)* | `55.50` · `1344.8` | `55.50` · `1344.8` | — | **CONTENTION CONFOUND:** `r4-minimal-core.md:52` admits shared GPU load (`50.91 ms` uncontended; **`44.98 ms`** idle on our GPU 0!) |
| **`law4-energy-v3`** | `readout-b898e928` | `bf16_channels_last` | `GPU 0` | `None` | `45.89` · `1147.2` | `45.89` · `12.56 J/st` | `45.58 ms` · **`13.12 J`** *(P1: `12.56 J`)* · `9.04 J` · `89.6 W` | **SPLICED:** `45.89 ms` from Forge + `12.56 J/st` from Phase-1 profile (vs `13.12 J/st` in Phase-2!) |
| **`law4-energy-v10`** | `readout-9599d9a2` | `bf16_channels_last` | `GPU 0` | **`True`** | **`56.35`** · `1243.4` | **`45.30`** · `11.68 J/st` | `45.30 ms` · `11.68 J` · **`8.45 J`** · `70.9 W` | **WRONG RECEIPT LINK + CONTENTION:** line 169 cites profiler `45.30 ms` while linking `readout-a0e3e7f6` (`disc_stab`, no `r256`); actual Forge `r256` was `contended=True` (`56.35 ms`). **LOWER dynamic `J/st` (`8.45 J`) than `unified-v12` (`8.68 J`)!** |
| **`t4-nash-v3`** | `readout-642d4c55` | `bf16_channels_last` | `GPU 0` | `None` | `39.68` · `1372.9` | `39.68` · `1372.9` | — | Verbatim match; still fastest on our idle GPU 0 (`35.17 ms`, `10.94 J/st`) |
| **`unified-v12`** | `readout-59898e1a` | `bf16_channels_last` | `GPU 0` | `False` | **`41.39`** · **`1243.4`** | **`39.76`** *(L193)* / **`39.22`** *(L169)* · `1243.9` · `11.37 J/st` | `39.22 ms` · `11.37 J` · **`8.68 J`** · **`68.2 W`** | **THREE-WAY DISCREPANCY + HARNESS SPLICING + LOWEST `idle W` (`68.2 W`)** |

### 2.2 Five Concrete Methodology & Comparability Confounds

1. **Precision Confound (`float32` NCHW vs. `bf16_channels_last` NHWC):** `configs/forge/ideas/rhbi-v12.json` omits `training_precision`, resolving to **`float32`**, while every subsequent champion from `empf-v8` onward sets **`bf16_channels_last`**. As our RTX A6000 control proves (Section 4), switching `rhbi-v12` to `bf16_channels_last` alone accounts for **`-27.5%` latency, `-41.9%` VRAM, and `-26.6%` energy**!
2. **Benchmark Harness & Batch-Reuse Confound (`scaling_benchmark.py` vs. `law4_optimizer_profiler.py`):**
   - Official Forge `dcgan_performance_r256` ([`experiments/forge/scaling_benchmark.py:348-356`](../../experiments/forge/scaling_benchmark.py#L348-L356)): `3` warmup + `10` measured steps, **fresh `torch.rand` batches allocated per step**, CPU-side `time.perf_counter()` around synchronized `step(collect_stats=False)`, `peak_allocated_mib = 1243.4 MiB`, `particle_cloud` (`sigma=0.0`, `256` particles).
   - Standalone NVML profiler ([`experiments/law4_optimizer_profiler.py:588-619`](../../experiments/law4_optimizer_profiler.py#L588-L619)): `5` warmup + `50` measured steps, **reuses a single pre-filled `(real_buf, real_g_buf)` pair across all 50 steps**, CUDA-event step timing + NVML energy over full 50-step wall loop, `peak_allocated_mib = 1219.9 MiB`.
   - `docs/four-laws.md` mixes numbers from both harnesses interchangeably (`39.76 ms` Forge + `11.37 J/st` profiler in line 193; `45.89 ms` Forge + `12.56 J/st` Phase-1 profiler for `law4-energy-v3`; `45.30 ms / 11.68 J/st` profiler for `law4-energy-v10`).
3. **Contention & Device-Index Confounds:** Historical `r256` readouts (`rhbi-v12` through `law4-energy-v3`) lack NVML contention telemetry (`contended=None`), mix `GPU 0` and `GPU 1`, and include acknowledged shared-GPU runs (`minimal-v2` `55.50 ms` shared vs `44.98 ms` idle). Newer Forge `dcgan_performance` readouts for `law4-energy-v10` (`56.35 ms`) and `law4-energy-v11` (`61.02 ms`) explicitly record **`contended=True`**, making them non-comparable to uncontended runs.
4. **Idle-Power Baseline Drift (`68.15 W` vs. `116.74 W`) & Dynamic-Energy Inversion:** In `law4-phase2-energy-benchmark.json`, static idle GPU board power before each run drifted across **`48.6 W`** (`68.15 W` for `unified-v12`, `70.92 W` for `law4-energy-v10`, `83.86 W` for `unified-v11-post612-fastpath`, `89.61 W` for `law4-energy-v3`, `114.8–116.7 W` for `frob-sgd`/`lion`/`muon`) due to thermal/P-state settle time! Subtracting idle power (`dynamic_joules_per_step`) reverses the `unified-v12` vs `law4-energy-v10` energy ordering: **`law4-energy-v10` draws `8.45 J/st` dynamic vs `unified-v12`'s `8.68 J/st` dynamic** (`unified-v12` wins total `J/st` `11.37` vs `11.68` solely because faster step time amortizes static board power and its idle baseline was `2.77 W` cooler!).
5. **Energy-per-Step (`J/step`) vs. True Energy-to-Quality (`Joules-to-PASS`):**

| Task (Quality Gate?) | `unified-v12` Official Forge `b51ab67c` (`J/st` · `J_1st` · `J_sust` · `J_tot`) | `unified-v12` Phase-4 JSON (`J_1st` / `J_sust`) | `unified-v11` Post-#612+#615 Phase-4 JSON (`J_1st` / `J_sust`) | `law4-energy-v10` Phase-2 JSON (`J_1st` / `J_sust`) | Audit Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`dcgan_performance_r256`** *(NO quality gate!)* | `500.4 J` tot (13 st, `J/st=None`) | **`11.37 J/st`** (`8.68 J` dyn) | `20.96 J/st` (`14.81 J` dyn) | `11.68 J/st` (**`8.45 J` dyn**) | **Mock Uniform noise — ZERO image-quality / FID energy-to-quality measurement exists!** |
| **`ring16_acquisition`** | `2.98 J/st` · **`646.7 J`** (217) · `846.4 J` (284) · `4,769 J` | **`462.5 J`** / **`605.3 J`** | `627.5 J` / `837.7 J` | `518.8 J` / `679.0 J` | **`+39.8%` gap** between official `b51ab67c` (`646.7 J`) and Phase-4 JSON (`462.5 J`, post-PR-#615). Both approximate `J_1st = step * avg_J_per_step` (not true cumulative NVML integral!). |
| **`grid100`** | `5.39 J/st` · `None` · `None` · `37,739 J` | **`4,513 J`** (1000) / **`9,026 J`** (2000) | `7,083 J` / `12,749 J` | `4,939 J` / `9,878 J` | `b51ab67c` leaves `J_1st`/`J_sust` `None`; Phase-4 JSON shows genuine `-36.3%` gain for `v12` vs `v11`. |
| **`staggered100`** | `6.52 J/st` · `None` · `None` · `45,618 J` | `10,743 J` (2250) / `15,518 J` (3250) | `9,995 J`* *(FAIL suf)* | **`8,681 J`** (1500) / **`14,468 J`** (2500) | **`law4-energy-v10` beats `unified-v12` by `-19.2%` on `J_1st` (`8,681` vs `10,743 J`)!** |
| **`grid100_mlp16`** | `c1ee5c3b`: `J_tot` recorded, `J_1st=None` | `12,781 J` (1250) / `23,006 J` (2250) | **`9,808 J`** (1000) / **`19,617 J`** (2000) | — | **`unified-v11` (+#612+#615) beats `unified-v12` by `-23.3%` on `J_1st` (`9,808` vs `12,781 J`)!** |
| **`five_word_joint_smoke`** | `2.29 J/st` · `1,909 J` (834) · `9,540 J` (4167) · `45,790 J` | **`1,399 J`** (834) | `6,602 J` (2501) | — | Genuine **`4.72×` energy-to-quality win** for `unified-v12` over `v11` (3× fewer steps + 1.56× faster/step). |
| **`model_ae_ddgan_gaussian`** | `4.53 J/st` · `6,137 J` (1355) · `8,021 J` (1771) · `11,323 J` | — | `∞` (FAIL) | `∞` (timeout) | `unified-v12` reaches PASS on A6000; times out (`∞`) on L4. |

---

## 3. Occam's Razor, Code-Bloat, Speedup Attribution & Unablated `v11`/`v12` Mechanisms

### 3.1 Code Complexity Audit (`docs/four-laws.md` Line 193 vs. Actual Repository AST)

| Candidate Package | Docs Line 193 Core LOC | Actual Direct Files | Actual Direct LOC | Code LOC (non-blank) | AST `if` | AST `IfExp` | Total Branches | Transitive Sibling Packages Imported |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`unified-v12` (effective direct)** | **`—`** *(omitted!)* | **10** | **`8,564`** | `7,793` | **`818`** | **`239`** | **`1,057`** | `_unified_v4`, `_unified`, `_t1_coupling`, `_t2_critic`, `_t3_dynamics`, `_t4_arch`, `_rhbi`, `_bispf` |
| *↳ `unified-v12` delta only (`_unified_v12/`)* | — | 2 | `1,488` | `1,370` | `170` | `71` | `241` | `_unified_v11`, `_unified_v4` |
| **`unified-v11` (current `origin/develop`)** | **`4,713`** *(stale `8caab7bde`!)* | **8** | **`7,076`** | `6,423` | **`648`** | **`168`** | **`816`** | Same 8 sibling packages |
| `unified-v10` (`_unified_v4` + `unified_v4.py`) | `4,311` | 7 | `4,313` | `3,918` | `376` | `101` | `477` | `_unified`, `_t1_coupling`, `_t3_dynamics`, `_t4_arch`, `_rhbi` |
| `cond-fiber-v1` (`_cond_fiber` + `_unified`) | `4,244` | 12 | `4,261` | `3,773` | `310` | `76` | `386` | `_t1_coupling`, `_t2_critic`, `_rhbi`, `_bispf` |
| `t4-nash-v3` | `3,016` | 7 | `3,028` | `2,725` | `264` | `61` | `325` | `_empf`, `_t2_critic` |
| `rhbi-v12` | `3,114` | 6 | `3,114` | `2,883` | `233` | `83` | `316` | `_bispf` |
| `empf-v8` | `1,972` | 8 | `1,972` | `1,734` | `166` | `30` | `196` | `_bispf` |
| **`minimal-v2`** | **`899`** | **5** | **`899`** | `839` | **`128`** | **`118`** | **`246`** | **ZERO sibling imports** |

**Key Occam Finding:** `docs/four-laws.md` frames the progression as an Occam's Razor distillation, yet `unified-v12` has **`8,564` direct LOC and `1,057` branches** — **`1.99×` the LOC of `unified-v10`**, **`1.82×` the stale `4,713` LOC quoted for `unified-v11`**, and **`9.53×` the LOC of `minimal-v2`**! Furthermore, **3 of the 5 epicycles proved redundant by Round-4 Occam's Razor (`r4-occam-ablation-razor.md`) are STILL PRESENT in `unified-v12`**:
1. **`d_z == 4` Radon–Hurwitz 4-tangent quaternion branch** (`particlegan/_unified_v11/prior.py:1547-1573`) — proved 100% redundant by Round-4 Arm 3 (`readout-44dbbe8a`), still executes 3 extra generator forwards/step on `d_z == 4` tasks!
2. **Case A `fwAD` saturation inverse-attenuation (`_fwad_generator_probe` / `w_sat_pt`)** (`particlegan/_unified_v11/prior.py:967-1083`) — proved redundant on `zero_tanh_rail_escape` by Round-4 Arm 8 (`readout-453cc0e8`).
3. **2D `_conformal_local_horizons` flags (`rgdt_conformal_horizon: true`, `t1c_multiscale_horizon: true`)** (`configs/forge/ideas/unified-v12.json`) — proved bit-for-bit redundant on 12/12 non-1D probes by Round-4 Arm 6 (`readout-edffe8ce`).
*(Only 4D Conv2d NS and the critic logit secant band were actually removed in `UnifiedV12Optimizer`.)*

### 3.2 Exact Speedup Attribution (`588.75 -> 130.13 -> 89.19 -> 41.39 / 36.61 ms/step`)

| Transition | Observed `r256` Latency Drop | Exact Verified Mechanism(s) & Attribution Share |
| :--- | :--- | :--- |
| **`rhbi-v12` (`588.75 ms`) -> `empf-v8` (`130.13 ms`)** | `-458.62 ms` (`4.52×`) | **(1) Precision confound (`FP32 -> BF16 CL`):** `-27.5%` latency / `-41.9%` VRAM alone (verified directly in Section 4). **(2) Algorithmic:** Replacing 2nd-order Sinkhorn geodesic chord + heavy DualNorm SVD/FP64 polar (subsequently cut `588.75 -> 319.05 ms` in PR #630) with online *e-means* + lighter Born–Infeld penalty + `max_dim >= 128` hybrid NS. |
| **`empf-v8` (`130.13 ms`) -> `unified-v11` (`89.19 ms`)** | `-40.94 ms` (`1.46×`) | **Eliminating 2nd-order `torch.autograd.grad(..., create_graph=True)`:** Round-4 Arm 7 (`readout-4b04c4bf` vs `readout-90580c7a`) isolates this causally at **`-43.35 ms/step` (`133.01 -> 89.66 ms`, `1.48×`)** and `-477.7 MiB` VRAM via live-endpoint reuse + separable BF16 micro-secants. |
| **`unified-v11` (`89.19 ms`) -> `law4-energy-v3` (`45.89 ms`)** | `-43.30 ms` (`1.94×`) | **Dropping 4D Conv2d 5-step Quintic NS + unfused per-param copies:** Phase-1 profile & our GPU 0 breakdown isolate `opt_d + opt_g` collapsing from **`39.81–45.59 ms` (`~51%`) -> `2.45–2.62 ms` (`~5–6%`)** via zero-state `torch._foreach_norm` + in-place Frobenius-MP (`~27 ms` from Conv2d NS, `~17 ms` from FP32 copies/masks/allocations). |
| **`law4-energy-v3` (`45.89 ms`) -> `unified-v12` (`41.39` Forge / `39.22` P4 / `36.61` idle A6000)** | `-4.50` to `-9.28 ms` (`1.11–1.25×`) | **Zero-sync pullback / penalty / optimizer fastpath (PR #611 + PR #615):** Eliminating `new_tensor` H2D copies, strided tensor views, GPU-resident EMAs, and `_SUPPORTS_ZEROSYNC_V4_TENSORS` cuts GPU-CPU sync stalls (`113 -> 12` on `grid100`; `~2–3` on `r256`). *(Note: `0` 2D Critic NS adds `0.00 ms` on 4D Conv2d `r256` — its gain occurs on 2D MLPs!)* |

### 3.3 Untested Component Interactions & Unablated Mechanisms in `unified-v11` / `unified-v12`

Because **all 18 Round-4 Occam knockouts were executed on `unified-v10`**, the following **7 major mechanisms** added in `unified-v11` and `unified-v12` lack matched destructive ablation studies on `unified-v12`:

| Mechanism Added Post-`v10` | Source File & Lines | Direct Single-Factor Destructive Knockout on `unified-v12`? | Untested Interaction Risks |
| :--- | :--- | :---: | :--- |
| 1. **Zero-`eigh` cyclic Jacobi conditional fiber bank** | `_unified_v11/bank.py` (`795` LOC) | **NO** | Interacts with $K=2$ triangle-inequality radius compensation (`_compute_two_obj_pyth_scales`) and conditional cooling `r_cond`; unclear whether simplified 384-pair ring-only bank (`minimal.py`) suffices once `unified-v12` optimizer is used. |
| 2. **Zero-`pinv` analytic `direction_blend` projection** | `_unified_v11/optimizer.py:336-417` | **NO** | Interacts with `p_dec = 0.9995` and `min(2.0, ...)` radius scaling on multi-objective AE-DDGAN and VAE/BiGAN tasks (`bytegan_bigan_dictionary` failed `0/256` exact recon!). |
| 3. **Scale-invariant $k=4$ density-ratio orphan gate (`3.52`, `1.30`)** | `_unified_v11/loss.py` | **NO** *(only `v10 -> v11` diff)* | Interacts with surviving `rgdt_conformal_horizon` / `t1c_multiscale_horizon` flags and generator-free 2D Critic NS on `t3_iso_*`. |
| 4. **Generator-free 2D Critic NS fallback (`_generator_free`)** | `_unified_v12/optimizer.py:456, 643` | **PARTIAL (indirect)** *(`law4-energy-v10` vs `v11` diff)* | Only tested on 2 `t3_iso_*` probes (`0.3375 -> 0.4723`, `0.9062 -> 0.9629`); never ablated jointly with `v12`'s CountSketch or PR #615 fastpath. |
| 5. **Prime-67 CountSketch Period-2 Flip-Bifurcation controller (`flip_thresh=0.45`)** | `_unified_v12/optimizer.py:57-74, 126-170` | **NO** | **Highest fragility risk:** Active when `ema_fg < 0.60 and s_therm >= 0.5`; specifically targets `grid100_res16_tanh`, yet combined `v12+PR#615` and L4 both STILL FAIL `grid100_res16_tanh`, and `unified-v12` is `-23.3%` worse on `grid100_mlp16` `J-to-PASS` than `unified-v11`! No ablation proving CountSketch beats a simpler scalar step-cosine or EMA damping. |
| 6. **Role-level $K=2$ Conditional-Fiber Triangle-Inequality radius (`min(2.0, ...)` + `p_dec=0.9995`)** | `_unified_v12/optimizer.py:216-278` | **NO** | Only active when `is_cond_fiber and _protected.shape[0] == 2` (essentially a 2-task surgical mechanism targeting `model_ae_ddgan_{gaussian,bimodal}_chain`!). Never ablated on `K >= 3` objectives or tested against a simple constant `1.4x` boost. |
| 7. **Zero-sync pullback & penalty fastpaths (`PR #611` + `PR #615`, `+2,587` added lines)** | `_unified_v11/{prior,penalty,loss}.py` | **NO** *(performance refactor, yet caused 1-ULP `grid100_res16_tanh` regression!)* | Proves the "bit-exact fastpath" is **not** bit-exact on `grid100_res16_tanh` (flipping `7/7 -> 6/7`) while adding `>2,300` LOC of maintenance/branch complexity. |

---

## 4. Controlled Local RTX A6000 Micro-Benchmark (Independent Verification)

Executed via `PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python scripts/four_laws_audit/p9_verify_receipts.py --profile-gpu --device cuda:0` on strictly idle `NVIDIA RTX A6000` (`GPU 0`, `util_before = (0, 0)` on all 7 DCGAN arms, PyTorch `2.15.0+cu132`, fresh batches per step, `4` warmup + `15` measured steps; read-only, zero Forge receipts mutated):

### 4.1 `dcgan_performance_r256` (`256×256` RGB, `B=16`, `23.4M` params, Idle GPU 0)

| Arm (`training_precision`) | Median (`ms/st`) | p95 (`ms/st`) | Peak Alloc (`MiB`) | Total `J/step` | Dynamic `J/step` | Idle Baseline (`W`) | Util (`before -> after`) | Key Verification Insight |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |
| **`t4-nash-v3`** (`bf16_cl`) | **`35.17`** | `36.19` | `1,587.0` | **`10.94`** | `6.70` | `120.1` | `(0,0)->(90,57)` | Fastest & lowest `J/step` on `r256` (zero NS, zero pullback). |
| **`unified-v12`** (`bf16_cl`) | **`36.61`** | `40.44` | **`1,123.8`** | **`11.20`** | `8.34` | `77.0` | `(0,0)->(92,61)` | **Reproduces sub-40 ms & `~11.2 J/st`** (`1.81×` faster & `-40.6%` J/st vs current `v11`). |
| **`minimal-v2`** (`bf16_cl`) | **`44.98`** | `48.47` | `1,352.9` | **`11.18`** | `1.59`* | `211.0`* | `(0,0)->(74,44)` | Confirms committed `55.50 ms` was shared-load degraded; matches `v12` `J/step` in `899` LOC! |
| **`unified-v11`** (`bf16_cl`, current) | `66.38` | `66.99` | `1,217.9` | `18.86` | `11.45` | `111.6` | `(0,0)->(94,66)` | Post-PR-#615 speedup over historical `89.19 ms`, still bottlenecked by Conv2d NS (`51.0%`). |
| **`empf-v8`** (`bf16_cl`) | `121.34` | `122.45` | `1,530.1` | `33.69` | `13.81` | `163.6` | `(0,0)->(94,54)` | 2nd-order `autograd.grad` penalty + DualNorm overhead. |
| **`rhbi-v12` (native `float32`)** | `319.05` | `323.59` | `4,253.7` | `87.24` | `32.58` | `172.0` | `(0,0)->(88,57)` | Already `1.85×` faster than stale `588.75 ms` post-PR-#630. |
| **`rhbi-v12` (matched `bf16_cl`)** | **`231.28`** | `282.00` | **`2,472.3`** | **`64.05`** | `38.43` | `108.4` | `(0,0)->(85,34)` | **Direct proof of precision confound: `-27.5%` latency, `-41.9%` VRAM, `-26.6%` energy from BF16 CL alone!** |

*\*Note: Even with a 1.0s inter-arm cooldown on an idle GPU (`util=(0,0)`), NVIDIA RTX A6000 thermal/boost P-state decay causes `idle_power_watts` to vary (`77–211 W`), independently confirming why raw `dynamic_joules_per_step` is sensitive to cooldown duration unless a $\ge 10\text{s}$ thermal settle is enforced.*

### 4.2 Synchronized CUDA-Event Component Breakdown (`r256`) & 16-Layer 2D MLP (`grid100_mlp16`)

| Component (`dcgan_performance_r256`) | `unified-v11` (`ms` · `%`) | `unified-v12` (`ms` · `%`) | Speedup (`v11 -> v12`) | Remaining Bottleneck Diagnosis |
| :--- | :---: | :---: | :---: | :--- |
| **`d_forward_backward`** | `18.73 ms` (`24.0%`) | **`18.69 ms` (`46.2%`)** | `1.00×` | Irreducible DCGAN critic backbone compute |
| **`g_forward_backward`** | `12.19 ms` (`15.6%`) | **`12.03 ms` (`29.8%`)** | `1.01×` | Irreducible DCGAN generator backbone compute |
| **`d_penalty`** *(Law 2 BF16 micro-secant)* | `5.35 ms` (`6.9%`) | **`5.37 ms` (`13.3%`)** | `1.00×` | **#1 remaining non-backbone bottleneck (`55%` of all non-backbone time!)** |
| **`g_loss_prior`** *(Law 1 + Law 3)* | `1.87 ms` (`2.4%`) | `1.83 ms` (`4.5%`) | `1.02×` | Minor on `r256` (`particle_cloud`); larger on 2D pullback tasks |
| **`opt_d_step`** *(Law 4 critic)* | `17.92 ms` (`23.0%`) | **`0.48 ms` (`1.2%`)** | **`37.5×`** | Conv2d NS + copy elimination complete |
| **`opt_g_step`** *(Law 4 generator)* | `21.89 ms` (`28.1%`) | **`1.98 ms` (`4.9%`)** | **`11.1×`** | Remaining `~1.98 ms` is generator secant probe + foreach overhead |
| **Total Instrumented Step** | **`78.05 ms`** (`100%`) | **`40.42 ms`** (`100%`) | **`1.93×`** | Backbone (`76.0%`) + `d_penalty` (`13.3%`) = **`89.3%`** of step! |

**2D 16-Layer Skip-Free MLP (`grid100_mlp16` topology, `B=128`, `N=256`, `cuda:0`):**
- `minimal` (`0` NS): `59.43 ms/step` (`p95=61.17 ms`), `7.24 J/step`.
- **`unified_v12`** (2D G NS only: `g_ns5 = 425` over 25 steps = `17` 2D layers/step; **`d_ns = 0`**): **`63.11 ms/step`** (`p95=64.55 ms`), **`9.11 J/step`** — **`-19.5%` faster and `-16.3%` lower energy than `unified_v11`**, at only `+6.2%` latency over zero-NS `minimal`.
- `unified_v11` (2D G + D NS): `78.42 ms/step` (`p95=81.08 ms`), `10.87 J/step`.

---

## 5. Claim-by-Claim Audit Ledger (10-Category Taxonomy & 8-Point Rubric)

### Claim P9-1: *"Every score, timing, VRAM figure, energy measurement, and failure in `docs/four-laws.md` Part II is grounded verbatim in committed Forge study reports and immutable `reports/forge/records/readout-*.json` receipts."*
- **Classification:** `Falsified claim`
- **1. Formulation:** [`docs/four-laws.md`](../../docs/four-laws.md#L3) asserts 100% verbatim grounding in immutable committed `reports/forge/records/readout-*.json` receipts.
- **2. Assumptions:** Requires every linked receipt file to exist on `origin/develop`, every claimed view score to have a committed receipt, and every numeric table entry to match its linked receipt without splicing or misattribution.
- **3. Supporting evidence:** `38/40` historical readout references (Rounds 1–4 up through `unified-v11` and `law4-energy-v3`) genuinely exist and verify verbatim.
- **4. Counterexamples:** (a) Both `unified-v12` receipt links (`readout-5002641a64ddaa91f1fc3b87.json`, `readout-731e752ff3c4f97e29a84145.json`) and `particlegan/unified_v12.py` are broken links (files do not exist anywhere in Git); (b) `18/18` Tier-3 endurance, `5/5` shift recovery, `6/7` combined `t4-arch`, and `39.76 ms / 1243.9 MiB` have zero committed `readout-*.json` receipts; (c) committed `readout-59898e1a` records `41.39 ms / 1243.4 MiB`, contradicting both `39.76 ms` (line 193) and `39.22 ms` (line 169); (d) `law4-energy-champion.md:131` misattributes `unified-v12`'s genuine `interp_img_patch_shift` failure (`interp_peak=0.4915, finite_fraction=1.0` in `b51ab67c`) to the `0.99999994` float32 scorer edge; (e) `law4-energy-v10` line 169 cites profiler `45.30 ms` while linking `disc_stab` `readout-a0e3e7f6`.
- **5. Minimal falsification experiment:** Run `python scripts/four_laws_audit/p9_verify_receipts.py` or `pytest tests/four_laws_audit/test_p9_receipt_integrity.py`.
- **6. Prior research:** Pineau et al. (2021), *Improving Reproducibility in Machine Learning Research* (JMLR — ML Reproducibility Checklist); Gundersen & Kjensmo (2018), *State of the Art: Reproducibility in Artificial Intelligence* (AAAI).
- **7. Computational verification:** Programmatically verified in `test_four_laws_links_and_missing_receipts_detected` and `test_committed_unified_v12_readouts_and_three_way_r256_discrepancy`.
- **8. Confidence & uncertainty:** **100% (deterministic file/JSON check).**

### Claim P9-2: *"The `r256` progression (`588.75 -> 130.13 -> 89.19 -> 39.76/39.22 ms`) reflects a clean, apples-to-apples algorithmic speedup across identical benchmark contracts."*
- **Classification:** `Falsified claim` *(true underlying speedup exists, but reported progression is confounded across precision, harness, contention, and idle power)*
- **1. Formulation:** Comparing `rhbi-v12` (`588.75 ms`, `3750.6 MiB`), `empf-v8` (`130.13 ms`, `1435.4 MiB`), `unified-v11` (`89.19 ms`, `1372.6 MiB`), `minimal-v2` (`55.50 ms`), `law4-energy-v10` (`45.30 ms`), and `unified-v12` (`39.76` / `39.22 ms`, `11.37 J/st`) on `r256` isolates algorithmic efficiency gains under identical conditions.
- **2. Assumptions:** Equal precision (`dtype` & memory format), equal benchmark harness (batch regeneration, warmup/measured steps), equal contention state (`contended=False`), equal thermal/idle power baseline, and current codebase revision.
- **3. Supporting evidence:** Under matched `bf16_channels_last` on our strictly idle RTX A6000 (`util=(0,0)`), `unified-v12` genuinely achieves **`3.31×`** speedup over `empf-v8` (`121.34 -> 36.61 ms`), **`1.81×`** over current `unified-v11` (`66.38 -> 36.61 ms`), and **`6.32×`** over precision-matched `rhbi-v12` (`231.28 -> 36.61 ms`).
- **4. Counterexamples:** (a) `rhbi-v12` ran in **`float32`**, not `bf16_channels_last` (switching `rhbi-v12` to `bf16_channels_last` cuts latency `-27.5%`, VRAM `-41.9%`, energy `-26.6%`); (b) `39.22 ms / 11.37 J/st` used the 50-step single-batch-reuse profiler (`1219.9 MiB`) rather than Forge's 10-step fresh-batch `dcgan_performance` (`41.39 ms / 1243.4 MiB`); (c) `minimal-v2` (`55.50 ms`) and `law4-energy-v10`/`v11` (`56.35`/`61.02 ms`) Forge receipts suffered GPU contention (`minimal-v2` runs `44.98 ms` idle); (d) `law4-energy-v10` beats `unified-v12` on `dynamic_joules_per_step` (`8.45` vs `8.68 J/st`).
- **5. Minimal falsification experiment:** Run `p9_verify_receipts.py --profile-gpu --device cuda:0` (Section 4), comparing native FP32 vs matched BF16 CL `rhbi-v12` and idle `minimal-v2` / `t4-nash-v3` / `unified-v12`.
- **6. Prior research:** Dehghani et al. (2022), *The Efficiency Misnomer* (ICLR — warning against conflating hardware/precision/implementation effects with algorithmic cost); Henderson et al. (2020), *Towards the Systematic Reporting of the Energy and Carbon Footprints of Machine Learning* (JMLR).
- **7. Computational verification:** Verified in `test_progression_table_precision_and_contention_confounds` and our Section 4 RTX A6000 benchmark.
- **8. Confidence & uncertainty:** **Very High (`>95%`).** Both the confounds and the true underlying `~1.8×–6.3×` algorithmic gains are directly verified.

### Claim P9-3: *"Eliminating 2nd-order `autograd.grad` and 4D Conv2d Newton–Schulz drives >90% of `unified-v12`'s wall-clock and energy gains on high-resolution DCGANs."*
- **Classification:** `Empirically supported causal mechanism`
- **1. Formulation:** On `256×256` RGB DCGANs under matched `bf16_channels_last`, replacing 2nd-order `autograd.grad` with 1st-order live-endpoint/micro-secant Born–Infeld and replacing 4D Conv2d 5-step Quintic NS + unfused per-param copies with zero-state `_foreach` Frobenius-MP reduces total step time from `~133 ms -> 89 ms -> 40–45 ms` and cuts optimizer share from `~51%` to `~5–6%`.
- **2. Assumptions:** Applies to standard 4D Conv2d/ConvTranspose2d DCGAN architectures (`23.4M` params, `B=16`) where backbone forward/backward takes `~30.7 ms` on RTX A6000. Does *not* imply image-quality convergence on real datasets (`r256` is mock Uniform noise).
- **3. Supporting evidence:** Round-4 Arm 7 (`133.01 -> 89.66 ms`), Phase-1/2 profiles (`opt_d+opt_g: 45.59 -> 2.62 ms`), and our independent idle RTX A6000 CUDA-event breakdown (`opt_d+opt_g: 39.81 ms [51.0%] -> 2.45 ms [6.1%]`, `16.2×` optimizer speedup, `18.86 -> 11.20 J/st`).
- **4. Counterexamples / Scope limits:** On 2D 16-layer skip-free MLPs (`grid100_mlp16`), 4D Conv2d NS was never active, so the speedup is much smaller (`78.42 -> 63.11 ms`, `-19.5%`, from dropping 2D Critic NS) — and `unified-v12` actually spends `+30.3%` MORE `J-to-PASS` than `unified-v11` on `grid100_mlp16` (`12,781 J` vs `9,808 J`) due to slower convergence (`1250` vs `1000` steps)!
- **5. Minimal falsification experiment:** Re-enable 4D Conv2d NS or 2nd-order `autograd.grad` on `unified-v12` and measure `r256` phase breakdown (immediately adds `+37 ms` or `+43 ms`).
- **6. Prior research:** Mescheder et al. (2018) on R1/R2 double-backward cost; Jordan et al. (2024) / Bernstein & Newhouse (2024) on Muon / Newton–Schulz matrix orthogonalization cost vs Frobenius normalization.
- **7. Computational verification:** Independently verified via our Section 4.2 CUDA-event phase breakdown on RTX A6000.
- **8. Confidence & uncertainty:** **Very High (`>95%`)** for per-step compute/energy on `r256`; **Low/Unmeasured** for energy-to-FID on real natural images.

### Claim P9-4: *"Lower `J/step` on `dcgan_performance_r256` (`11.37 J/step`) establishes energy-to-quality optimality."*
- **Classification:** `Falsified claim`
- **1. Formulation:** Treating `r256` `11.37 J/step` as an energy-to-quality Pareto certificate conflates per-step power draw on mock noise with total energy required to reach target generative quality (`Joules-to-PASS`).
- **2. Assumptions:** Would require `steps-to-quality` to be constant across optimizers and `r256` mock-noise steps to proxy real image convergence.
- **3. Supporting evidence:** `unified-v12` *does* achieve strong `J-to-PASS` on `ring16` (`462.5 J` Phase-4 / `646.7 J` Forge), `grid100` (`4,513 J`), and `five_word_joint_smoke` (`1,399 J`, `4.72×` less than `v11`).
- **4. Counterexamples:** (a) `law4-pure-sgd` has the *lowest* `r256` `J/step` (`11.10 J/st`), yet spends `∞ J` on `grid100` and `+57%` more energy on `ring16` (`973 J` transient FAIL vs `619 J` PASS); (b) `unified-v12` has lower `r256` `J/step` than `unified-v11` (`11.37` vs `20.96 J/st`) yet spends **more `J-to-PASS` on `grid100_mlp16` (`12,781 J` vs `9,808 J`)** and **more `J-to-PASS` on `staggered100` than `law4-energy-v10` (`10,743 J` vs `8,681 J`)**; (c) `dcgan_performance_r256` has NO quality gate at all, and Forge's `joules_to_first_pass` is a linear average approximation (`step * J_tot / N`), not a true integrated energy reading at `first_pass_step`.
- **5. Minimal falsification experiment:** Compare `J/step` ranking vs `J-to-PASS` ranking across `law4-pure-sgd`, `law4-energy-v10`, `unified-v11`, and `unified-v12` on `grid100_mlp16` and `staggered100` (already inverted in `law4-phase2-energy-benchmark.json`).
- **6. Prior research:** Patterson et al. (2021), *Carbon Emissions and Large Neural Network Training*; Dahl et al. (2023), *Benchmarking Neural Network Training Algorithms* (MLCommons AlgoPerf — emphasizing **time/energy-to-target-quality**, never time-per-step).
- **7. Computational verification:** Verified directly from `law4-phase2-energy-benchmark.json` and `readout-b51ab67c` in Section 2.2.
- **8. Confidence & uncertainty:** **Very High (`100%`).**

### Claim P9-5: *"`unified-v12` represents an Occam's-Razor-pruned minimal realization of the Four Laws whose added mechanisms are causally verified."*
- **Classification:** `Falsified claim` *(for `unified-v12`; `minimal-v2` at `899` LOC is a genuine compact heuristic core, though it passes only `10/17` Tier 1)*
- **1. Formulation:** `unified-v12` eliminates redundant epicycles identified in Round 4 and retains only causally verified minimal mechanisms.
- **2. Assumptions:** Assumes LOC/branch count decreased or stabilized after Round 4, proven redundant epicycles were removed, and newly added `v11`/`v12` mechanisms underwent destructive ablation.
- **3. Supporting evidence:** `UnifiedV12Optimizer` did remove 4D Conv2d NS and the critic logit secant band, and Round-4 Occam knockouts on `unified-v10` strongly established causal necessity of Laguerre dual balancing (`7/15` without), 1st-order `fwAD`/secant (`1.48×` faster + higher pass rate), Pythagorean LTV (`9/15`), `delta_g_max` (`6/16` modes without), and 2D NS on `grid100_mlp16` (`prec=0.024` without).
- **4. Counterexamples:** (a) Direct LOC exploded from `4,313` (`unified-v10`) -> `4,713` (orig `v11`) -> `7,076` (current `v11`) -> **`8,564` LOC / `1,057` branches** (`unified-v12`) + 8 transitive sibling packages; (b) **3 proven Round-4 redundant epicycles survive in `unified-v12`** (`d_z == 4` Radon–Hurwitz in `prior.py:1547`, `w_sat_pt`/`_fwad_generator_probe` in `prior.py:967-1083`, `rgdt_conformal_horizon` enabled in `unified-v12.json`); (c) **6 of 7 major `v11`/`v12` mechanisms have ZERO single-factor destructive knockouts** (especially CountSketch flip feedback and $K=2$ triangle-inequality radius compensation).
- **5. Minimal falsification experiment:** Run single-factor destructive knockouts on `unified-v12` removing: (i) CountSketch flip feedback (`_update_period2_flip_feedback`), (ii) $K=2$ triangle-inequality radius (`_compute_two_obj_pyth_scales`), (iii) surviving `d_z == 4` Radon–Hurwitz + `w_sat_pt` + `rgdt_conformal_horizon`, and measure full `discriminator_stability` + `t4-arch` + `FUSQP-v1`.
- **6. Prior research:** Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems* (NeurIPS); Lipton & Steinhardt (2019), *Troubling Trends in Machine Learning Scholarship* (ACM Queue — warning against post-hoc mechanism accretion disguised as principled theory).
- **7. Computational verification:** Verified in `test_code_complexity_and_surviving_occam_epicycles` (`8,564` LOC, `1,057` branches, all 3 surviving epicycles confirmed `True`).
- **8. Confidence & uncertainty:** **Very High (`100%` on LOC, epicycles, and missing ablation receipts).**

### Claim P9-6: *"The `d_penalty` first-order Born–Infeld micro-secant (`5.37 ms`, `13.3%` of step) is the primary remaining non-backbone computational bottleneck on high-resolution Conv2d GANs."*
- **Classification:** `Empirically supported causal mechanism` *(architecture/hardware-specific to `dcgan_performance_r256` on RTX A6000)*
- **1. Formulation:** Once 4D Conv2d NS and unfused copies are removed in `unified-v12`, model forward/backward accounts for `76.0%` (`30.72 ms`) of step time on `r256`, and `d_penalty` (`5.37 ms`, `13.3%`) accounts for `55.4%` of all remaining non-backbone overhead (`9.70 ms`).
- **2. Assumptions:** Standard `dcgan_performance_r256` (`256×256`, `B=16`, `bf16_channels_last`, RTX A6000).
- **3. Supporting evidence:** Independently measured in our Section 4.2 CUDA-event breakdown (`d_penalty = 5.37 ms / 40.42 ms = 13.28%`, vs `opt_g = 1.98 ms [4.89%]`, `g_loss_prior = 1.83 ms [4.53%]`, `opt_d = 0.48 ms [1.18%]`), matching `docs/four-laws.md` line 113's `~15.8%` estimate from Phase 1.
- **4. Counterexamples:** On 2D vector/MLP tasks (`grid100`, `grid100_mlp16`), pullback prior regularization (`12` CUDA syncs/step) and 2D generator 5-step Quintic NS dominate instead of Conv2d `d_penalty`.
- **5. Minimal falsification experiment:** Evaluate amortized (`reg_every = 2` or `4`, lazy R1-style) first-order micro-secants vs per-step `d_penalty` on `r256` and `bcap_convolution_images` / real-image FID to test if `~2.7–4.0 ms/step` can be reclaimed without destabilizing Law 2.
- **6. Prior research:** Karras et al. (2020), *StyleGAN2* (CVPR — lazy regularization `reg_every = 16` reducing critic penalty overhead by `>10×`).
- **7. Computational verification:** Directly measured via `p9_verify_receipts.py --profile-gpu --device cuda:0`.
- **8. Confidence & uncertainty:** **High (`>90%`)** on `r256` RTX A6000; amortization quality impact remains an `Unresolved question` until tested on real-image FID.

---

## 6. Concrete Proposed Repairs for `docs/four-laws.md` (Without Editing It Here)

Per strict file ownership, we do NOT modify `docs/four-laws.md` or any historical receipt. We recommend the synthesis/integration track apply the following exact factual corrections:

1. **Fix the 3 Broken Links:**
   - Line 169: replace `readout-5002641a64ddaa91f1fc3b87.json` -> `readout-b51ab67c4d97526186e23651.json`, and `readout-731e752ff3c4f97e29a84145.json` -> `readout-c1ee5c3bb7ac4f432532da63.json` (plus cite `readout-59898e1a77c4b179c9aebbc1.json` and `readout-624964f4e46793593bc147e4.json`).
   - Line 175: replace `../particlegan/unified_v12.py` -> `../particlegan/_unified_v12/optimizer.py` (noting it inherits `particlegan/_unified_v11/`).
2. **Resolve the Three-Way `r256` Discrepancy & Uncommitted Claims:**
   - In the line-193 table, report `unified-v12`'s committed Forge `dcgan_performance` figure (**`41.39` · `1243.4`** from `readout-59898e1a`), and move the standalone 50-step NVML profiler figure (`39.22 ms` · `11.37 J/st` total / `8.68 J/st` dynamic) to a clearly separated energy footnote/table rather than splicing harnesses inside one cell.
   - Mark `18/18` Tier-3 endurance, `5/5` shift recovery, and `6/7` combined `t4-arch` as *uncommitted Phase-4 profiler/local observations* until official `readout-*.json` files are committed.
3. **Fill Core LOC Honestly & Flag Comparability Confounds:**
   - Replace `—` for `unified-v12` with **`8,564`** (`1,488` delta + `7,076` `_unified_v11`) and update `unified-v11` from stale `4,713` (`8caab7bde`) to **`7,076` (current)** (retaining `4,713` as its original `8caab7bde` note).
   - Add an explicit `training_precision` column (`float32` for `rhbi-v12` vs `bf16_channels_last` for all newer champions) and contention flag (`shared` on `minimal-v2`, `contended=True` on `law4-energy-v10`/`v11` Forge receipts) to the Cross-Champion table.
4. **Correct `interp_img_patch_shift` Attribution:** State transparently that while `unified-v11` hit the `0.99999994` float32 scorer edge, `unified-v12` on RTX A6000 (`readout-b51ab67c`) had `finite_fraction = 1.0` and genuinely failed numerically (`interp_peak = 0.4915 < 0.78`).
5. **Run Missing `unified-v12` Destructive Ablations:** Before declaring `unified-v12` minimal, delete the 3 surviving Round-4 redundant epicycles (`d_z == 4` Radon–Hurwitz, `w_sat_pt`, 2D `_conformal_local_horizons`) and run single-factor knockouts on CountSketch flip control and $K=2$ triangle-inequality radius compensation.
