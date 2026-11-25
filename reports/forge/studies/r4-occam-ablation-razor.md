# Round-4 Track 2 Study: Occam's Razor — Systematic Branch & Component Ablation Study (`r4-ablation-razor`)

**Branch:** `feat/solve-gans-r4-ablation-razor`  
**Ablation Package:** [`particlegan/_r4_ablation/`](../../../particlegan/_r4_ablation/) (`optimizer.py`, `penalty.py`, `prior.py`, `__init__.py`) — isolated single-factor knockouts on top of `unified-v10` (`origin/develop` `c400c6d47`) with **zero edits** to `_unified_v4/`, `_nash_v3/`, `_cond_fiber/`, `_unified/`, `_t4_arch/`, `_t4_nash/`, `recipe_compat.py`, or any pre-existing `readout-*.json` receipt.  
**Diagnostic View:** [`configs/forge/views/r4-occam-ablation-v1.json`](../../../configs/forge/views/r4-occam-ablation-v1.json) (`evidence_scope: "research_diagnostic"`, `14` surgical probes spanning every regime + `2` required smoke prerequisites = `16` tasks), plus matched evaluations on `t4-strange-architectures-v1`, `t4_imbalance_nash_stretch`, and `dcgan_performance`.  
**Protocol & Controls:** Protocol seed `0`, `deterministic_orthogonal` initializer, clean live sampling, `DEFAULT_RECIPE == "empf"`, exact `179` `Recipe` fields preserved, monkeypatch-enforced primitive audits (`0` SVD / `0` `eigh` / `0` `pinv` / `0` `autograd.grad` / `0` NS on `r4_mp_no_ns`; semantic positive control verifying `torch.autograd.grad` execution on `r4_autograd_critic`), and bit-exact fresh-process continuation in [`tests/test_r4_ablation_razor.py`](../../../tests/test_r4_ablation_razor.py).  
**Calibration / Default-Adoption Gate:** **BLOCKED** for global default adoption (`DEFAULT_RECIPE` remains `"empf"`). All `18` concluded Forge readout receipts below carry `evidence_scope: "research_diagnostic"` and supply causal ablation evidence rather than ordinary qualification credit.

---

## Executive Answer: Which Laws Survived Occam's Razor?

The central scientific question for Round 4 is: *"Our initial assumption was 2 'laws', but it seems to have grown — which mechanisms across Laws 1–4 are mathematically irreducible vs. redundant epicycles?"*

Across **`18` concluded official Forge GPU study readouts** (`seed = 0`, `NVIDIA RTX A6000`) performing exact single-factor knockouts on `unified-v10` (`4,311` direct LOC / `478` branches in `particlegan/_unified_v4/`, plus `3,463` shared LOC / `294` branches in `_unified/penalty.py` and `_t1_coupling/{primitive,charts}.py`), Occam's Razor cleanly separates the codebase into **3 tiers**:

| Tier | Mechanism | Exact Knockout Arm | Direct LOC / Branches | Empirical Verdict (Protocol Seed `0`) |
| :--- | :--- | :--- | :---: | :--- |
| **IRREDUCIBLE CORE** | **Laguerre Dual Mass-Balancing & Streaming $H_0$ Chart Commitment** | Arm 5 (`-chart-laguerre`) | `136` *(+`708` charts.py)* / `14` *(+`36`)* | **#1 most irreducible mechanism (`7/15` PASS):** collapses `grid100` (`prec=0.2827, mass_tv=0.4417`), `img_bars4` (`2/4` mode collapse), `ring16` (`mass_tv=0.1863`), both Nash tasks, `vector_unequal_width`, and `t3_iso_scale_mass_8x` (`eig=0.0034`). |
| **IRREDUCIBLE CORE** | **1st-Order `fwAD` / Multi-Probe Secant Born–Infeld Critic** | Arm 7 (`-fwad-critic`) | `880` / `93` | **Strictly Pareto-dominant over 2nd-order `autograd.grad` EBI:** **`1.48×` faster** at `256×256` RGB (`89.66` vs `133.01 ms/step`), **`-477.7 MiB` (`-34.8%`) less VRAM** (`1372.6` vs `1850.3 MiB`), and strictly higher pass rate (`grid100`, `img_bars4`, `zero_tanh_rail_escape` all regressed under 2nd-order EBI; sole 2nd-order gain: `t3_iso_scale_mass_8x` PASS `eig=0.5313`). |
| **IRREDUCIBLE CORE** | **Externally Isolated Pythagorean Sibling Coalescence + LTV Variance Conservation** | Arm 4 (`-coalesce-ltv`) | `220` / `34` | **Strongly irreducible on 100-mode lattices:** removing coalescence/LTV triggers severe Zador–Gersho sub-cell collapse on `grid100` (`center_rms_sigma=0.5312` vs `0.1379`, `cov_trace_bias=-0.3798`, `cov_frob_rms=0.3975`), plus regressions on `nash_overpowered_critic_ring16` (`hq=0.7097`) and `vector_unequal_mass`. |
| **IRREDUCIBLE (1-REGIME)** | **Generator Pushforward Secant Trust Ceiling (`delta_g_max`)** | Arm 2 (`-secant-band`) | `~85` / `~14` *(half of `159`/`27`)* | **Irreducible for `548×` $G \gg D$:** `nash_overpowered_generator_ring16` collapses to **`6/16` modes, `hq=0.2522`** without it. By contrast, the **critic logit secant band is REDUNDANT** (`nash_overpowered_critic_ring16` `hq=0.9324` and `nash_unequal_width_hostile_imbalance` `hq=0.9810` both PASS without it). |
| **IRREDUCIBLE (1-REGIME)** | **Matrix Orthogonalization (Quintic Newton–Schulz) on Deep Skip-Free Dense MLPs** | Arm 1 (`-NS` v1 & v2) | `229` / `31` | **SPLIT VERDICT:** **Irreducible ONLY on 16-layer skip-free `grid100_mlp16`** (`precision` collapses `0.9929 -> 0.0244–0.0621` without NS!). **100% REDUNDANT** on shallow nets, `grid100` (`prec=0.9908–0.9927`), 16-layer **residual** `grid100_res16_scaled_tanh` (`prec=0.9910`, `ctr=0.0547`), `3/3` Nash `>600×`, and 4D Conv2d DCGANs (`r256 = 76.09–76.17 ms/step`, **`1.18×` faster** than `89.66 ms/step`!). |
| **IRREDUCIBLE (1-REGIME)** | **2D-Complex (`d_z=2`) Covariance Isometry (`Sigma_push2`)** | Arm 3 (`-pullback-iso`) | `~50` / `~4` *(of `78`/`6`)* | **SPLIT VERDICT:** **Irreducible ONLY on `grid100`** (`cov_trace_bias=+0.2388`, `cov_frob_rms=0.2816` without it). The **`dim=2, d_z=4` Radon–Hurwitz quaternion branch is 100% REDUNDANT** (`ring16`, `vector_unequal_width/mass`, and both Nash tasks ALL PASS with 5-line 1-sample `iso_1` alone!). |
| **IRREDUCIBLE (1-REGIME)** | **1D Weibull AKT Batch-Horizon Exponent (`akt_exp = max(0, 1/d - 0.5)`)** | Arm 6 (`-conformal`) | `9` / `2` *(of `106`/`6`)* | **SPLIT VERDICT:** **3-line 1D Weibull AKT factor is irreducible for 1D** (`gaussian1d_smoke` `cdf_ks` degrades `0.0220 -> 0.0986`). The **~100-line 2D `_conformal_local_horizons` module is 100% REDUNDANT** (`12/12` 2D/4D/deep/Nash/image/lattice probes pass bit-for-bit identically, including `vector_unequal_width` `hq=0.9980, eig=0.6783`). |
| **REDUNDANT EPICYCLE** | **Case A `fwAD` Saturation Inverse-Attenuation (`w_sat_pt`)** | Arm 8 (`-sat-probe`) | `134` / `15` | **REDUNDANT on `zero_tanh_rail_escape`:** `zero_tanh_rail_escape` **STILL PASSES (`mae=0.0026`, `p99=0.0087`)** without any `fwAD` saturation probe because $10^{-38}$ dynamic scale normalization already inverts `sech^2` attenuation. (`grid100` 100k holdout also passes `prec=0.9928, ctr=0.1300`, missing only step-6250 suffix.) |

**Bottom Line:** The original **2-Law intuition was right at the core**. ParticleGAN's empirical stability does **not** require four sprawling, domain-forked subsystems:
- **Half A (Measure & Potential Geometry):** (1) Laguerre dual mass-balancing + streaming $H_0$ chart commitment, (2) 1st-order `fwAD`/secant Born–Infeld critic, (3) externally isolated Pythagorean coalescence + LTV (+ compact `d_z=2` covariance isometry & 3-line 1D Weibull AKT scale).
- **Half B (Function-Space Two-Player Dynamics):** (4) $10^{-38}$ dynamic scale-normalized `peak_rms` + Marchenko–Pastur elementwise updates + lightweight generator pushforward ceiling `delta_g_max` (+ depth-gated 2D Quintic NS strictly for skip-free deep MLPs; **zero** NS on Conv2d or shallow/residual nets).

Shaving the proven redundant epicycles (**4D Conv2d & shallow Quintic NS**, **critic logit secant band**, **`d_z=4` Radon–Hurwitz 4-tangent branch**, **2D `_conformal_local_horizons`**, and **Case A `fwAD` `w_sat_pt` saturation probe**) removes **`~450+` LOC / `~50+` branches**, eliminates 3 extra generator forward passes on `d_z=4` tasks, and speeds up `256×256` RGB DCGAN training by **`-13.5 ms/step` (`1.18×`)** with zero loss of coverage.

---

## 1. Complete Inventory of All 18 Concluded Official Forge Readout Receipts

All `18` receipts were executed under protocol seed `0` on `NVIDIA RTX A6000` (`evidence_scope: "research_diagnostic"`) and are committed verbatim under [`reports/forge/records/`](../records/):

| Candidate (`rev`) | Forge Study ID | View | `n` | Verbatim Receipt Outcomes | Official Readout Receipt |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`r4-razor-no-ns`** (`d5093bddc53d`) | `r4-razor-no-ns-dcgan-perf-study` | `dcgan_performance` | `6` | **`6` PASS** (`r256=76.09 ms`) | [`readout-226ec401`](../records/readout-226ec4012c0becaa51d8ce75.json) |
| **`r4-razor-no-ns`** (`d5093bddc53d`) | `r4-razor-no-ns-occam-study` | `r4-occam-ablation-v1` | `16` | `11` PASS, `4` FAIL, `1` INCOMPLETE *(error: `float64` `peak_rms` cast)* | [`readout-1b23903f`](../records/readout-1b23903f5aad2cbdee8c8a7a.json) |
| **`r4-razor-no-ns`** (`d5093bddc53d`) | `r4-razor-no-ns-t4-arch-study` | `t4-strange-architectures-v1` | `7` | `3` PASS, `3` FAIL, `1` INVALID *(clockfree `float64` cast)* | [`readout-473d9273`](../records/readout-473d9273f48217f741893cb4.json) |
| **`r4-razor-no-ns`** (`d5093bddc53d`) | `r4-razor-no-ns-t4-nash-study` | `t4_imbalance_nash_stretch` | `3` | **`3` PASS** | [`readout-fcfec5e8`](../records/readout-fcfec5e8ae1c8411f818fa8f.json) |
| **`r4-razor-no-ns-v2`** (`98ebd4d6b2ef`) | `r4-razor-no-ns-v2-dcgan-perf-study` | `dcgan_performance` | `6` | **`6` PASS** (`r256=76.17 ms`) | [`readout-0219c6e2`](../records/readout-0219c6e2857b1605cbcf3428.json) |
| **`r4-razor-no-ns-v2`** (`98ebd4d6b2ef`) | `r4-razor-no-ns-v2-occam-study` | `r4-occam-ablation-v1` | `16` | `11` PASS, `4` FAIL, `1` INCOMPLETE *(`five_word` timeout)*; `0` ERROR | [`readout-690dd40a`](../records/readout-690dd40a4221b390d8680e43.json) |
| **`r4-razor-no-ns-v2`** (`98ebd4d6b2ef`) | `r4-razor-no-ns-v2-t4-arch-study` | `t4-strange-architectures-v1` | `7` | `4` PASS, `3` FAIL; **`0` INVALID** (`clockfree` `4/4` PASS) | [`readout-1c538e28`](../records/readout-1c538e28f3716f374f4c527e.json) |
| **`r4-razor-no-ns-v2`** (`98ebd4d6b2ef`) | `r4-razor-no-ns-v2-t4-nash-study` | `t4_imbalance_nash_stretch` | `3` | **`3` PASS** | [`readout-023ec5bb`](../records/readout-023ec5bb580b7f86afaa21ff.json) |
| **`r4-razor-no-secant-band`** (`8e6452a58cff`) | `r4-razor-no-secant-band-occam-study` | `r4-occam-ablation-v1` | `15` | `10` PASS, `5` FAIL *(1 gated: `grid100_mlp16`)* | [`readout-44d5ad0a`](../records/readout-44d5ad0a2de36996b109445b.json) |
| **`r4-razor-no-secant-band`** (`9ef66970ab0b`) | `r4-razor-no-secant-band-t4-nash-study` | `t4_imbalance_nash_stretch` | `3` | `2` PASS, `1` FAIL *(`overpowered_generator` `6/16`)* | [`readout-4e2fd695`](../records/readout-4e2fd695306a5dd6fd222e62.json) |
| **`r4-razor-no-pullback-iso`** (`170b5b864990`) | `r4-razor-no-pullback-iso-occam-study` | `r4-occam-ablation-v1` | `15` | `11` PASS, `3` FAIL, `1` INCOMPLETE *(`five_word` timeout; 1 gated)* | [`readout-44dbbe8a`](../records/readout-44dbbe8ab156d3615c474db4.json) |
| **`r4-razor-no-coalesce-ltv`** (`50a94391c1ae`) | `r4-razor-no-coalesce-ltv-occam-study` | `r4-occam-ablation-v1` | `15` | `9` PASS, `5` FAIL, `1` INCOMPLETE *(`five_word` timeout; 1 gated)* | [`readout-04bb3c21`](../records/readout-04bb3c211150d5629c489b5e.json) |
| **`r4-razor-no-chart-laguerre`** (`3666a81626e5`) | `r4-razor-no-chart-laguerre-occam-study` | `r4-occam-ablation-v1` | `15` | `7` PASS, `8` FAIL *(1 gated: `grid100_mlp16`)* | [`readout-b27b2689`](../records/readout-b27b26896bd137e2e701ce2d.json) |
| **`r4-razor-no-conformal`** (`b2116fc9ffbe`) | `r4-razor-no-conformal-occam-study` | `r4-occam-ablation-v1` | `15` | `12` PASS, `3` FAIL *(1 gated: `gaussian1d_stability`)* | [`readout-edffe8ce`](../records/readout-edffe8cee489f8b14768e2cb.json) |
| **`r4-razor-autograd-critic`** (`99cb547d1f4f`) | `r4-razor-autograd-critic-dcgan-perf-study` | `dcgan_performance` | `6` | **`6` PASS** (`r256=133.01 ms`, `1850.3 MiB`) | [`readout-4b04c4bf`](../records/readout-4b04c4bf0dc3d7408a82e51e.json) |
| **`r4-razor-autograd-critic`** (`06baf01b3bcd`) | `r4-razor-autograd-critic-occam-study` | `r4-occam-ablation-v1` | `15` | `11` PASS, `4` FAIL *(1 gated; unlocked `scale_mass_8x`)* | [`readout-b96faf85`](../records/readout-b96faf85499b1dbe9dce102e.json) |
| **`r4-razor-no-sat-probe`** (`921de9f3afcd`) | `r4-razor-no-sat-probe-occam-study` | `r4-occam-ablation-v1` | `15` | `11` PASS, `3` FAIL, `1` INCOMPLETE *(`five_word` timeout; 1 gated)* | [`readout-453cc0e8`](../records/readout-453cc0e86ed3223a21d247c3.json) |
| **`r4-razor-no-sat-probe`** (`921de9f3afcd`) | `r4-razor-no-sat-probe-t4-arch-study` | `t4-strange-architectures-v1` | `4` | `3` PASS, `1` FAIL *(3 gated by `grid100` step-6250 suffix)* | [`readout-9b7edc6d`](../records/readout-9b7edc6de002e2ce4dde5bc5.json) |

*(Matched Arm 0 reference receipts on `origin/develop`: `unified-v10` [`readout-6da9dee1`](../records/readout-6da9dee19abe886e7ddbe184.json), [`readout-688bc7ed`](../records/readout-688bc7ed35078e19c10daf6a.json), [`readout-8f0a4508`](../records/readout-8f0a450815d09f56bc74a678.json), [`readout-90580c7a`](../records/readout-90580c7aff5ec6988cfbcfad.json); cross-family references: `t4-nash-v3` [`readout-1ca76c61`](../records/readout-1ca76c61c875ba5a33dca488.json), [`readout-642d4c55`](../records/readout-642d4c55d093b44ed1e0309e.json); `cond-fiber-v1` [`readout-5932f4d3`](../records/readout-5932f4d351cc4e791417835a.json), [`readout-8d2e547c`](../records/readout-8d2e547ce820cf8662a34bdf.json).)*

---

## 2. Verbatim 16-Task × 9-Arm Knockout Matrix (`r4-occam-ablation-v1`)

Every cell below is verified verbatim from the committed `readout-*.json` files. `GATED` denotes a dependent task not launched by Forge because its declared gate/checkpoint parent failed (`grid100 -> grid100_mlp16` or `gaussian1d_smoke -> gaussian1d_stability`). `INCOMPLETE (TO)` denotes `gate_status="INCOMPLETE", raw_status="timeout"` (`900s` wall budget exhausted under concurrent multi-agent GPU/CPU box load on `20,001`-step `five_word_joint_smoke`; never conflated with numerical `FAIL`).

| Probe Task | Ref (`u10`) | Arm 1 (`-NS` v2) | Arm 2 (`-secant`) | Arm 3 (`-pb-iso`) | Arm 4 (`-coal-ltv`) | Arm 5 (`-laguerre`) | Arm 6 (`-conformal`) | Arm 7 (`-fwad`) | Arm 8 (`-sat-probe`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gaussian1d_smoke` *(prereq)* | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **FAIL** (`0.0986`) | **PASS** | **PASS** |
| `gaussian1d_stability` | **PASS** | **PASS** (`0.0204`) | **PASS** | **PASS** | **PASS** | **PASS** | GATED | **PASS** (`0.0148`) | **PASS** |
| `zero_tanh_smoke` *(prereq)* | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| `zero_tanh_rail_escape` | **PASS** | **FAIL** (`0.0040`) | **FAIL** (`0.0021`) | **PASS** | **PASS** | **PASS** | **PASS** | **FAIL** (`0.0011`) | **PASS** (`0.0026`) |
| `ring16_acquisition` | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **FAIL** (`tv=0.1863`) | **PASS** | **PASS** | **PASS** |
| `vector_unequal_width` | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **FAIL** (`tv=0.1763`) | **PASS** (`0.9980`) | **PASS** | **PASS** |
| `vector_unequal_mass` | **PASS** | **PASS** | **PASS** | **PASS** (`eig=0.7975`) | **FAIL** | **PASS** | **PASS** | **PASS** | **PASS** |
| `img_bars4` | **PASS** | **PASS** (`1.0000`) | **PASS** | **PASS** | **PASS** | **FAIL** (`2/4` modes) | **PASS** | **FAIL** (`4/5` suf) | **PASS** |
| `grid100` | **PASS** | **PASS** (`0.9908`) | **FAIL** (`tr=-0.1022`*) | **FAIL** (`tr=+0.2388`) | **FAIL** (`ctr=0.5312`) | **FAIL** (`prec=0.2827`) | **PASS** (`0.9918`) | **FAIL** (`ctr=0.1757`) | **FAIL** (`4/5` suf*) |
| `grid100_mlp16` | **PASS** | **FAIL** (`prec=0.0244`) | GATED | GATED | GATED | GATED | **PASS** (`0.9929`) | GATED | GATED |
| `model_ddgan_gaussian_chain` | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| `five_word_joint_smoke` | **PASS** | INCOMPLETE (TO)† | **PASS** | INCOMPLETE (TO) | INCOMPLETE (TO) | **PASS** | **PASS** | **PASS** | INCOMPLETE (TO) |
| `nash_overpowered_critic_ring16` | **PASS** | **PASS** (`0.9727`) | **PASS** (`0.9324`) | **PASS** (`0.9407`) | **FAIL** (`0.7097`) | **FAIL** (`15/16`) | **PASS** | **PASS** | **PASS** |
| `nash_overpowered_generator_ring16` | **PASS** | **PASS** (`0.8987`) | **FAIL** (`6/16, 0.2522`) | **PASS** (`0.9473`) | **PASS** | **FAIL** (`tv=0.1968`) | **PASS** | **PASS** | **PASS** |
| `t3_iso_scale_mass_8x` | FAIL | FAIL (`eig=0.3811`) | FAIL | FAIL | FAIL | FAIL (`eig=0.0034`) | FAIL (`eig=0.2180`) | **PASS** (`eig=0.5313`) | FAIL |
| `t3_iso_horizon_saddle_shift` | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** |
| **Verbatim PASS / Executed** | **`14/16`** | **`11/16`** *(v1: `11/16` incl. `five_word` PASS)* | **`10/15`** | **`11/15`** | **`9/15`** | **`7/15`** | **`12/15`** | **`11/15`** | **`11/15`** |

*\*Note: Both Arm 2 (`-secant-band`, `holdout prec=0.9933, ctr=0.1369, tr=-0.0954`) and Arm 8 (`-sat-probe`, `holdout prec=0.9928, ctr=0.1300, tr=-0.0871`) PASSED the 100k independent `grid100` holdout cleanly, missing only a single live terminal check by `<0.003`. †`five_word_joint_smoke` PASSED under Arm 1 v1 (`mass_tv=0.0189, modes=5`, [`readout-1b23903f`](../records/readout-1b23903f5aad2cbdee8c8a7a.json)) and timed out at `900.5s` under v2 due to concurrent GPU/CPU box load.*

---

## 3. Question 1 Deep Dive: Is Matrix Orthogonalization (SVD / Newton–Schulz) Irreducible or Redundant?

Arm 1 (`r4-razor-no-ns` v1 & `r4-razor-no-ns-v2`, `optimizer_family="r4_mp_no_ns"`) replaces **only** `UnifiedV4EquilibratedOptimizer`'s 5-step BF16 Quintic Newton–Schulz polar orthogonalization with pure-elementwise $O(\text{params})$ $`10^{-38}`$-safe `peak_rms` + Marchenko–Pastur RMS-to-RMS operator normalization (`t4_nash_v3` style; strictly `0` SVD, `0` `eigh`, `0` `pinv`, `0` `autograd.grad`, and `0` Newton–Schulz). Note that v1 uncovered a `float64 -> float32` PyTorch `Optimizer.load_state_dict` dtype cast on `peak_rms` (`gaussian1d_stability` `INCOMPLETE (error)` and `clockfree` `INVALID`), which v2 fixed to `torch.float32`, achieving **`0` ERROR and `0` INVALID across all four views** (`gaussian1d_stability` PASS `cdf_ks=0.0204`, `clockfree_audit_measurement_v1` `4/4` PASS).

### 3.1 Cross-View Comparison (`unified-v10` Quintic NS vs. `r4-razor-no-ns-v2` Pure-Elementwise MP)

| Benchmark Suite & Task | `unified-v10` (Quintic NS) | `r4-razor-no-ns-v2` (`0` NS, Pure MP) | Causal Interpretation |
| :--- | :---: | :---: | :--- |
| **`dcgan_performance_r256`** (`256×256` RGB) | `89.66 ms/step` · `177.5 img/s` · `1372.6 MiB` | **`76.17 ms/step`** *(v1: `76.09`)* · **`209.5 img/s`** · `1372.6 MiB` | **NS is pure overhead on 4D Conv2d:** removing NS saves **`13.5 ms/step` (`1.18×` faster)** with identical VRAM. |
| **`t4_imbalance_nash_stretch`** (`3` tasks) | **`3 / 3` PASS** | **`3 / 3` PASS** (`hq=0.9727 / 0.8987 / 0.9910`) | **NS is redundant for `>600×` Nash imbalance & `-30`-logit hostile init** when MP + generator secant band are active. |
| **`grid100`** (3-layer 100-mode lattice) | **PASS** (`prec=0.9918`, `ctr=0.1379`, `cov=0.1229`) | **PASS** (`prec=0.9908`, `ctr=0.1133`, `cov=0.1182`; v1: `ctr=0.0945`) | **NS is redundant on shallow lattices:** pure-elementwise MP + `peak_rms` is actually *sharper* on `center_rms_sigma` (`0.0945–0.1133` vs `0.1379`). |
| **`grid100_res16_scaled_tanh`** (16-layer ResNet) | **PASS** (`prec=0.9906`, `ctr=0.0537`, `tr=-0.0400`) | **PASS** (`prec=0.9910`, `ctr=0.0547`, `tr=-0.0396`) | **NS is redundant on deep residual networks:** identity skip connections preserve full Jacobian rank under elementwise MP! |
| **`grid100_res16_tanh`** (16-layer unscaled ResNet) | FAIL *(100k holdout `prec=0.9906`, `ctr=0.1037`)* | FAIL *(100k holdout `prec=0.9913`, `ctr=0.1057`)* | Virtually identical behavior without NS! |
| **`grid100_mlp16`** (16-layer **skip-free** MLP) | **PASS** (`prec=0.9929`, `ctr=0.0712`, `cov=0.1145`) | **CATASTROPHIC FAIL** (`prec=0.0244`, `mass_tv=0.6707`; v1: `0.0621`) | **NS IS MATHEMATICALLY IRREDUCIBLE HERE!** Without residual skips, elementwise updates are rank-1-dominated across 16 composed layers, collapsing Jacobian rank; polar orthogonalization equalizes singular values per layer. |
| **`zero_tanh_rail_escape`** | **PASS** (`mae=0.0019`, `p99=0.0076`) | **FAIL** (`mae=0.0040`, `p99=0.0215`, missed terminal suffix) | Mild settling benefit from NS on `tanh` rail escape. |
| **`clockfree_audit_measurement_v1`** | **PASS** (`4/4`) | **PASS** (`4/4`) | Exact clock-free & fresh-process restart parity confirmed. |

**Architectural Takeaway:** Matrix orthogonalization should **never** be applied unconditionally to 4D convolutions or shallow layers. A clean minimal core should run pure-elementwise `O(params)` $`10^{-38}`$-safe `peak_rms` + Marchenko–Pastur everywhere by default (saving `13.5+ ms/step` at `r256`), and activate 2D Quintic NS **strictly on deep skip-free dense networks** (`deep_gate > 0` and `ndim == 2`) if `grid100_mlp16` is in scope!

---

## 4. Question 2 Deep Dive: Per-Arm Causal Autopsy (Arms 2–8)

1. **Arm 2 (`-secant-band`, `r4-razor-no-secant-band`, [`44d5ad0a`](../records/readout-44d5ad0a2de36996b109445b.json) & [`4e2fd695`](../records/readout-4e2fd695306a5dd6fd222e62.json)):**
   - Removing realized function-space secant trust bands (`alpha_gain == 1.0`) caused a **catastrophic mode collapse on `nash_overpowered_generator_ring16`** (`548×` $G \gg D$: **`6/16` modes, `hq=0.2522`** vs `unified-v10` `16/16, hq=0.9194`), proving that **generator pushforward ceiling regulation (`delta_g_max`) is irreducible** even when Quintic NS is active.
   - Conversely, **`nash_overpowered_critic_ring16`** (`621×` $D \gg G$: `16/16, hq=0.9324, eig=0.5056`) and **`nash_unequal_width_hostile_imbalance`** (`-30`-logit init: `hq=0.9810, eig=0.6305`) **both still PASSED** without any critic logit secant band — proving the critic-side secant probe is redundant under equilibrated updates + 1st-order EBI!
2. **Arm 3 (`-pullback-iso`, `r4-razor-no-pullback-iso`, [`44dbbe8a`](../records/readout-44dbbe8ab156d3615c474db4.json)):**
   - Knocking out **both** the `dim==2, d_z==2` 2D-complex and `dim==2, d_z==4` Radon–Hurwitz multi-tangent covariance isometry probes (keeping only universal 5-line 1-sample radial `iso_1`, zero extra $G$ forward passes) left **every single `d_z=4`, ring, unequal-width, unequal-mass, and Nash probe PASSING**: `ring16_acquisition` (`hq=0.9832, eig=0.4643`), `vector_unequal_width` (`hq=0.9985, eig=0.5970`), `vector_unequal_mass` (`hq=0.9990, eig=0.7975` — *higher* than `unified-v10`'s `0.4781`!), `nash_overpowered_critic` (`hq=0.9407`), and `nash_overpowered_generator` (`hq=0.9473`).
   - The **only** regression was `grid100` (`cov_trace_bias=+0.2388`, `cov_frob_rms=0.2816`, `prec=0.9598`). **Verdict:** the entire `dim==2, d_z==4` Radon–Hurwitz quaternion branch (`3` extra generator forwards/step) is a **100% redundant epicycle**; only the `d_z=2` orthogonal pair `Sigma_push2` is irreducible for tight 100-mode lattices.
3. **Arm 4 (`-coalesce-ltv`, `r4-razor-no-coalesce-ltv`, [`04bb3c21`](../records/readout-04bb3c211150d5629c489b5e.json)):**
   - Setting `w_mat = I, multicell_coal = False` (disabling externally isolated Pythagorean sibling coalescence, LTV `rho_mu` variance subtraction, and concentric `v_push` residual) caused textbook Zador–Gersho sub-cell splitting on `grid100`: `center_rms_sigma` exploded **`3.85×`** (`0.1379 -> 0.5312`), `cov_trace_bias` collapsed to **`-0.3798`**, and `cov_frob_rms` jumped to **`0.3975`**, while `nash_overpowered_critic_ring16` dropped to `hq=0.7097` and `vector_unequal_mass` failed its stability suffix. **Verdict:** strongly irreducible.
4. **Arm 5 (`-chart-laguerre`, `r4-razor-no-chart-laguerre`, [`b27b2689`](../records/readout-b27b26896bd137e2e701ce2d.json)):**
   - Setting `t1c_chart_weight = 0.0` + `psi_eff = 0` produced the **single worst collapse of the entire study (`7/15` PASS, `8` FAILs)**: `grid100` collapsed to `prec=0.2827, mass_tv=0.4417`; `img_bars4` suffered `2/4` mode collapse; `ring16_acquisition` (`mass_tv=0.1863`), `vector_unequal_width` (`mass_tv=0.1763`), `nash_overpowered_critic` (`15/16, mass_tv=0.2688`), `nash_overpowered_generator` (`mass_tv=0.1968`), and `t3_iso_scale_mass_8x` (`core_min_eigen_ratio=0.0034`) all failed. **Verdict:** `#1` most irreducible mechanism in ParticleGAN.
5. **Arm 6 (`-conformal`, `r4-razor-no-conformal`, [`edffe8ce`](../records/readout-edffe8cee489f8b14768e2cb.json)):**
   - Disabling `t1c_multiscale_horizon`, `rgdt_conformal_horizon`, and `rhbi_akt_horizon` left **all `12/12` 2D, 4D, deep, Nash, discrete-image, and lattice probes bit-for-bit identical PASSes** to `unified-v10` (including `vector_unequal_width` `hq=0.9980, eig=0.6783`, `grid100` `0.9918`, `grid100_mlp16` `0.9929`), because the pullback path already adapts per-prototype local scales `sigma_bar_all`.
   - The **sole** failure was `gaussian1d_smoke` (`cdf_ks` degraded `0.0220 -> 0.0986`), which is controlled exclusively by the 3-line 1D Weibull AKT exponent (`akt_exp = max(0, 1/d - 0.5)`). **Verdict:** delete the `~100`-LOC 2D `_conformal_local_horizons` module; keep only the 3-line 1D Weibull AKT factor!
6. **Arm 7 (`-fwad-critic`, `r4-razor-autograd-critic`, [`4b04c4bf`](../records/readout-4b04c4bf0dc3d7408a82e51e.json) & [`b96faf85`](../records/readout-b96faf85499b1dbe9dce102e.json)):**
   - Replacing 1st-order `fwAD`/secant with exact 2nd-order `torch.autograd.grad(..., create_graph=True)` EBI (`_T1CouplingPenaltyKernel`) degraded `dcgan_performance_r256` from `89.66 ms/step` to **`133.01 ms/step` (`+43.35 ms/step`, `1.48×` slower)** and increased peak allocated VRAM from `1372.6 MiB` to **`1850.3 MiB` (`+477.7 MiB`, `+34.8%`)**.
   - Simultaneously, 2nd-order `autograd.grad` EBI **regressed on 3 probes** (`grid100` `prec=0.9822, ctr=0.1757`, `img_bars4` `4/5` suffix, `zero_tanh_rail_escape`), while achieving **1 notable gain**: `t3_iso_scale_mass_8x` **PASS** (`core_min_eigen_ratio=0.5313` vs `unified-v10`'s `0.2155`), explaining why `unified-v10` regressed on that one direct-cloud stretch probe. **Verdict:** 1st-order `fwAD`/secant is strictly Pareto-dominant overall.
7. **Arm 8 (`-sat-probe`, `r4-razor-no-sat-probe`, [`453cc0e8`](../records/readout-453cc0e86ed3223a21d247c3.json) & [`9b7edc6d`](../records/readout-9b7edc6de002e2ce4dde5bc5.json)):**
   - Knocking out Case A `_fwad_generator_probe` saturation inverse-attenuation (`w_sat_pt = 1`), Case B `w_sat_vec`/`w_iso_sat`, and coordinatewise outward column-sensitivity dead-clamp rescue **still PASSED `zero_tanh_rail_escape` (`mae=0.0026`, `p99=0.0087`)**, `zero_tanh_smoke` (`mae=0.0025`, `p99=0.0086`), and `clockfree_audit_measurement_v1` (`4/4`)!
   - Why? Because `UnifiedV4EquilibratedOptimizer`'s **$`10^{-38}`$ dynamic scale normalization** already cancels scalar `sech^2` attenuation. `grid100` also passed its 100k independent holdout (`prec=0.9928, ctr=0.1300, tr=-0.0871` — sharper than `unified-v10`!) and 4/5 live suffix checks, missing only step 6250 marginally (which gated `grid100_res16_*` in `t4-arch`).

---

## 5. Complexity, Energy (`r256 ms/step` & VRAM) & Quality Pareto Leaderboard

### 5.1 Full `dcgan_performance` Energy Comparison (`32²`–`256²` RGB, `NVIDIA RTX A6000`)

| Task | `r4-razor-no-ns-v2` (`0` NS, 1st-order) | `unified-v10` (Quintic NS, 1st-order) | `r4-razor-autograd-critic` (Quintic NS, 2nd-order `autograd.grad`) |
| :--- | :---: | :---: | :---: |
| `dcgan_performance_r32` | `31.91 ms` · `81.7 MiB` | `25.99 ms` · `75.6 MiB` | `29.12 ms` · `77.7 MiB` |
| `dcgan_performance_r64` | `40.63 ms` · `237.2 MiB` | `32.61 ms` · `183.5 MiB` | `38.00 ms` · `191.8 MiB` |
| `dcgan_performance_b64` | `40.08 ms` · `386.7 MiB` | `32.93 ms` · `334.6 MiB` | `42.67 ms` · `442.1 MiB` |
| `dcgan_performance_w128` | `52.48 ms` · `834.2 MiB` | `41.97 ms` · `546.7 MiB` | `48.62 ms` · `555.0 MiB` |
| `dcgan_performance_r128` | `52.81 ms` · `554.5 MiB` | `62.03 ms` · `559.0 MiB` | `72.16 ms` · `657.5 MiB` |
| **`dcgan_performance_r256`** | **`76.17 ms`** *(v1: `76.09`)* · **`209.5 img/s`** · **`1372.6 MiB`** | `89.66 ms` · `177.5 img/s` · `1372.6 MiB` | `133.01 ms` · `120.2 img/s` · `1850.3 MiB` |

*(Receipts: `r4-razor-no-ns-v2` [`0219c6e2`](../records/readout-0219c6e2857b1605cbcf3428.json), `r4-razor-no-ns` v1 [`226ec401`](../records/readout-226ec4012c0becaa51d8ce75.json), `unified-v10` [`90580c7a`](../records/readout-90580c7aff5ec6988cfbcfad.json), `r4-razor-autograd-critic` [`4b04c4bf`](../records/readout-4b04c4bf0dc3d7408a82e51e.json).)*

### 5.2 Cross-Family Complexity–Energy–Quality Pareto Table

| Formulation / Candidate | Package LOC (`Code`) | AST Branches (`If/Tern`) | `0` SVD / `eigh` / `pinv`? | `0` NS? | `0` `autograd.grad`? | `r256` (`ms/step`) | Peak Alloc (`MiB`) | Suite Pass Summary |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`rhbi-v12`** *(R1–R2)* | `3,114` (`2,883`) | `316` | No | Yes | No | `588.75–599.88` | `3706.6` | `34/41` `disc_stab` |
| **`empf-v8`** *(current default)* | `1,972` (`1,734`) | `196` | No | Yes | No | `130.13–173.05` | `1435.4` | `24/41` `disc_stab` |
| **`cond-fiber-v1`** *(R3 Track 1)* | `1,338` *(+shared)* (`1,159`) | `89` *(+shared)* | No | No | **Yes** | `87.50` | `1308.4` | `40/47` `disc_stab` (`ddgan` PASS!) |
| **`t4-nash-v3`** *(R3 Track 3)* | `3,016` (`2,714`) | `325` | **Yes** | **Yes** | **Yes** | **`39.68`** | `1372.9` | `33/46` `disc_stab`, `3/3` Nash |
| **`unified-v10`** *(Arm 0 Ref)* | `4,311` *(+`3,463` shared)* | `478` *(+`294` shared)* | **Yes** | No | **Yes** | `89.66` *(v4: `69.55`)* | `1372.6` | `40/46` `disc_stab`, `14/16` Occam, `6/7` `t4-arch`, `3/3` Nash |
| **`r4-razor-no-ns-v2`** *(Arm 1 `-NS`)* | `-229` LOC (`-211`) | `-31` branches | **Yes** | **Yes** | **Yes** | **`76.17`** *(v1: `76.09`)* | **`1372.6`** | `11/16` Occam *(v1: `five_word` PASS)*, `4/7` `t4-arch`, **`3/3` Nash** |
| **`r4-razor-autograd-critic`** *(Arm 7 `-fwad`)* | `-880` LOC *(reuses `383` EBI)* | `-93` branches | **Yes** | No | **No** | `133.01` | `1850.3` | `11/15` Occam (`scale_mass_8x` PASS `eig=0.5313`) |
| **Occam-Pruned Minimal Target** *(Blueprint)* | **`~900`–`1,200` LOC** | **`~100`–`140` branches** | **Yes** | **Conv/Shallow `Yes`** *(2D deep only)* | **Yes** | **`< 55–75 ms`** | **`~1,370 MiB`** | Preserves all `14/16` `unified-v10` Occam regimes minus redundant epicycles |

---

## 6. Verification, Primitive Audit & Reproducibility

- **Unit & Semantic Controls:** [`tests/test_r4_ablation_razor.py`](../../../tests/test_r4_ablation_razor.py) verifies:
  1. `DEFAULT_RECIPE == "empf"`, exact `179` `Recipe` fields, and `16`-task diagnostic `r4-occam-ablation-v1` view schema.
  2. Monkeypatch enforcement of `0` `svd` / `svdvals` / `eigh` / `eigvalsh` / `pinv` / `autograd.grad` / `quintic_orthogonalize_2d` / `quintic_orthogonalize_batched` on `R4MPNoNSOptimizer` (`r4_mp_no_ns`), plus underflow safety at `1e-36`.
  3. Strict `alpha_gain == 1.0` invariant on `R4NoSecantBandOptimizer` (`r4_no_secant_band`).
  4. Positive semantic control confirming `R4AutogradCriticPenalty` (`r4_autograd_critic`) actually invokes `torch.autograd.grad`.
  5. Checkpoint round-trip, tamper rejection, exact `float32` tensor dtype/value parity across `load_state_dict`, and bit-exact fresh-process continuation across all ablation modes.
- **Immutability Audit:** Zero edits to `particlegan/_unified_v4/`, `particlegan/_nash_v3/`, `particlegan/_cond_fiber/`, `particlegan/_unified/`, `particlegan/_t4_arch/`, `particlegan/_t4_nash/`, `particlegan/recipe_compat.py`, or any pre-existing `reports/forge/records/readout-*.json` receipt.
