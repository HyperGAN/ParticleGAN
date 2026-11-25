# Axiom Four-Laws Adversarial Head-to-Head Benchmark

**Scope & Provenance:** Deterministic (`seed=0`, `deterministic_orthogonal_` initialization, matched architectures, batch sequences, prior capacities, and step budgets) head-to-head stress benchmark comparing final **`axiom`** (`axiom-v4`, commit `59a860fa0`) against **`unified_v12`**, **`unified_v13`**, **`minimal`**, and **`rhbi`** across the five Four-Laws audit controls where prior ParticleGAN families exhibit documented structural failures (see [Four-Laws Integration Report](../four-laws-audit/integration-report.md), [Axiom Design Spec](design-spec.md), [Gap Analysis](gap-analysis.md), and [Axiom Forge Study](axiom-study.md)).

- **Branch / Evaluated Commit:** `docs/axiom-v4-four-laws-bench` from `origin/feat/axiom-family` (`59a860fa0`, `axiom-v4` complete, `2,399 / 2,500` core LOC)
- **Benchmark Driver:** [`scripts/axiom_four_laws_benchmark.py`](../../scripts/axiom_four_laws_benchmark.py)
- **Hardware & Runtime:** Physical RTX A6000 `cuda:0` (`CUDA_VISIBLE_DEVICES=0`, keeping `cuda:1` idle), PyTorch `2.15.0+cu132`, total wall time `294.70 s`
- **Reproduction Command:**
  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
      scripts/axiom_four_laws_benchmark.py --device cuda:0
  ```

> [!NOTE]
> All measurements below are single-seed (`seed=0`) adversarial stress controls executed through each recipe's public API without altering any baseline's formula. They verify structural Law 1–4 invariants and identify concrete engineering gaps; see [Axiom Forge Study](axiom-study.md) for the full 9-study RTX A6000 Forge evaluation battery (including the `four-laws-adherence-v1` view).

---

## Executive Four-Laws Leaderboard

| Audit Control / Law | Key Metric (Target) | **`axiom`** (`v4`) | `unified_v12` | `unified_v13` | `minimal` | `rhbi` |
|---|---|---:|---:|---:|---:|---:|
| **1A. Law 4.2 (P5): 16L Critic Swap (`ring8` B1)** | Modes / `hq` (`8/8`, `>= 0.90`) | **8/8 / 0.9417** (**PASS**) | 8/8 / 0.5000 (FAIL) | 8/8 / 0.5000 (FAIL) | 0/8 / 0.0073 (FAIL) | 0/8 / 0.0000 (FAIL) |
| **1C. Law 4.2 (P0): Rank-1 `128x128` NS (`s=0`)** | $\sigma_2 / \sigma_1$ (`< 1e-3`), $\sigma_1$ (`~1.0`) | **1.29e-5 / 1.124** (**PASS**) | 0.4484 / 6.823 (FAIL) | 0.4484 / 6.823 (FAIL) | N/A (0 NS) | exact polar |
| **2A. Law 4.4 (P1): 24-Decade Scale Invariance** | `max_diff` on `[1e-12, 1e12]` (`< 1e-5`) | **1.61e-6** (**PASS**) | 0.326 (FAIL) | 0.326 (FAIL) | 0.176 (FAIL) | 0.606 (FAIL) |
| **2B–D. Law 4.4 (P1): Rail + Spike + Chatter** | `rail(1e6)` / `spike(1e-6)` / `chatter` | **1.95e-5 / 1.95e-5 / 0.001** (**PASS**) | 0.015 / 3.02 / 1.00 (FAIL) | 0.015 / 3.02 / 1.00 (FAIL) | 0.0 / 2.93 / 1.00 (FAIL) | 6.85 / 3.54 / 1.00 (FAIL) |
| **3A. Law 4.1 (P4): Verified Trust (`tanh` / `cubic` / `16L`)** | Worst `p95 / d_max` (`<= 1.0x`) | **0.54x / 0.00x / 0.54x** (**PASS**) | 5.00x / 10.24x / 1.15x (FAIL) | 5.00x / 10.24x / 1.15x (FAIL) | 5.00x / 301.0x / 1.01x (FAIL) | 5.00x / 0.00x / 4.16x (FAIL) |
| **3B–C. Law 4.3 (P4): `coord_scale` + `1.6%` Tail** | `1e-4` ratio (`1.0x`) / Tail captured | **1.000x / True** (**PASS**) | 500.0x / False (FAIL) | 500.0x / False (FAIL) | 500.0x / False (FAIL) | N/A / False (FAIL) |
| **4A. Law 2 (P6): Below-$\kappa$ Unbiasedness (`d=24`)** | `MC / Exact` loss (`~1.0x`) & grad cos | **1.07x / cos 0.990** (**PASS**) | 1423.3x / cos 0.993 (FAIL) | 1423.3x / cos 0.993 (FAIL) | 0.65x / cos 0.536 (FAIL) | 0.004x / vanishes (FAIL) |
| **4B. Law 2 (P6): Dim-Indep $\mu_D > 0$ (`2D` vs `256^2`)** | `eff_r256 / eff_2d` (`~1.0x`) | **1.06x** (`2.65e-3`) (**PASS**) | 3.60e-3 (`278x` dilute, FAIL) | 3.60e-3 (`278x` dilute, FAIL) | 1953x blowup (FAIL) | 7.09e-3 (`3.5e-15`, FAIL) |
| **5A–C. Law 1 (P7): Mass, LTV, Gauge & $B_f < K$** | `wf_err` / `LTV(M,S)` / `B<K` | **5.96e-8 / (1.00, 1.00) / 36/36** (**PASS**) | 0.119 / (0.27, 0.81) / 0/36 (FAIL) | 0.119 / (0.27, 0.81) / 0/36 (FAIL) | 0.119 / false-10x / $\psi=0.8$ (FAIL) | N/A (no atlas) |
| **5D. Runtime: Conv2d `64x64` vs MLP `grid100`** | `ms/step` (`syncs/step`) | **21.48 ms (4.0)** / 44.37 ms (35.9) | 22.88 ms (5.0) / 32.82 ms (10) | 21.79 ms (5.0) / 32.73 ms (10) | 27.96 ms (6.0) / 57.29 ms (28) | 91.99 ms (75) / **25.45 ms (4)** |

**Summary Score (Structural Law Controls 1A, 1C, 2A–D, 3A–C, 4A–B, 5A–C):**
- **`axiom` (`v4`, `59a860fa0`): 9 / 9 PASS**, plus **#1 mode coverage (`23/100`) & `hq` (`0.0560`)** on the 500-step 16-layer skip-free `grid100_mlp16` early-budget stress snapshot (`3.3x–4.6x` more modes than `unified_v12` `5/100` / `unified_v13` `7/100`), and **#1 fastest & fewest-sync** Conv2d `GANTrainer` step (`21.48 ms/step`, `4.0 syncs/step`).
- **`unified_v12` / `unified_v13`: 0 / 9 PASS** across these nine adversarial failure controls.
- **`minimal`: 0 / 9 PASS**.
- **`rhbi`: 0 / 9 PASS** (except exact polar reference in 1C).

---

## Suite 1 — Law 4.2 (P5 Role-Agnostic 2D NS & P0 FP32 Tikhonov NS)

### 1A. Regime B1 — Deep 16-Layer Skip-Free Critic Swap (`ring8`, Pure Adversarial RpGAN + Law 2 BI $\beta=2.0$, 300 Steps)

Architecture: `3L` skip-free generator (`SimpleMLPGenerator(2, 128, 3, 2)`) vs `16L` skip-free critic (`SimpleMLPDiscriminator(2, 128, 16, fourier=0)`), `512` MoG particles, `bsz=256`. Every preset runs its unmodified `recipe.make_optimizers(G, D, prior=prior)`.

| Preset | Modes / 8 | `hq` | `mass_tv` | `d_acc_gauss` | `d_acc_valley` | `ns_G` | `ns_D` | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`axiom`** | **8/8** | **0.9417** | 0.0229 | 0.656 | 0.607 | 1200 | **4800** | 12.0 |
| `unified_v12` | 8/8 | 0.5000 | **0.0190** | **0.734** | **0.625** | 1200 | 0 | 10.3 |
| `unified_v13` | 8/8 | 0.5000 | **0.0190** | **0.734** | **0.625** | 1200 | 0 | 10.0 |
| `minimal` | 0/8 | 0.0073 | 0.8750 | 0.491 | 0.549 | 0 | 0 | 22.5 |
| `rhbi` | 0/8 | 0.0000 | 0.5901 | 0.566 | 0.492 | 0 | 0 | 40.7 |

**Takeaway:** `unified_v12`/`v13` hardcode NS exclusively to the generator (`ns_G=1200`, `ns_D=0`), starving the 16-layer skip-free critic and stalling generator quality at `hq=0.5000`. `axiom-v4`'s strictly role-agnostic 5-step FP32 Tikhonov NS on all 2D dense and isomorphic ungrouped `4D 1x1` Conv2d layers (`md >= 2`, keeping `0` NS on spatial `kh>1`/`kw>1` Conv2d) rescues the 16L critic (`ns_D=4800`), lifting `hq` by **+44.2pp** (`0.9417`) while lowering `mass_tv` to `0.0229` (down from `0.0354` in `v3` and `0.0735` in `v1`).

### 1B. Regime A — Deep 16-Layer Skip-Free Generator (`grid100_mlp16`, Real `GANTrainer`, 500 Steps)

Architecture: `16L` skip-free generator (`SimpleMLPGenerator(2, 128, 16, 2)`) + `3L` critic (`fourier=0`), `2048` MoG particles, `bsz=256`, 500 steps (early-budget stress snapshot; official Forge `grid100_mlp16` runs longer).

| Preset | Modes / 100 | `hq` | `ns_G` | `ns_D` | Wall (s) |
|---|---:|---:|---:|---:|---:|
| **`axiom`** | **23/100** | **0.0560** | 8500 | 1500 | 27.4 |
| `unified_v12` | 5/100 | 0.0511 | 8500 | 0 | 23.7 |
| `unified_v13` | 7/100 | 0.0356 | 8500 | 0 | **23.5** |
| `minimal` | 3/100 | 0.0196 | 0 | 0 | 58.3 |
| `rhbi` | 0/100 | 0.0000 | 0 | 0 | 36.4 |

**Takeaway:** At the 500-step early snapshot, `axiom-v4` leads all five families in both mode coverage (**`23/100`**, `3.3x–4.6x` higher than `unified_v13` `7/100` and `unified_v12` `5/100`) and `hq` (`0.0560`). Compared with `axiom-v3`'s faster early 500-step ramp (`100/100`), `axiom-v4`'s `V4E` `dec=0.02` + non-2D `s_eq_base=0.05` trades slower initial 500-step acquisition on skip-free `16L` MLP for substantially cleaner settling and zero-floor 2D spectral conditioning on long-horizon Forge tasks (see [Axiom Forge Study](axiom-study.md) for the full-budget `grid100_mlp16` readout).

### 1C. Healthy `r256` Conv2d Zero-NS Guard & P0 Rank-1 `128x128` NS Spectral Leakage

- **Healthy `DCGANCritic(256, 64, 512)` (`bf16` channels-last, 5 steps):** `ns_D = 0` for `axiom` (and all baselines), confirming zero Newton–Schulz overhead on healthy spatial Conv2d architectures while `4D 1x1` ungrouped Conv2d preserves exact `0.000e+00` single-step parity with `2D Linear`.
- **P0 Rank-1 `128x128` Matrix ($G = u v^T$) on CUDA:**

| Orthogonalizer | $\sigma_1$ (Target `~1.0`) | $\sigma_2$ | Leakage $\sigma_2 / \sigma_1$ |
|---|---:|---:|---:|
| **`axiom` `fp32_tikhonov_ns_2d` (`s=0.0`)** | **1.124** | **1.45e-5** | **1.29e-5** (**PASS**) |
| **`axiom` `fp32_tikhonov_ns_2d` (`s=0.05`)** | **0.858** | **5.27e-6** | **6.14e-6** (**PASS**) |
| `unified_v12`/`v13` BF16 `quintic_orthogonalize_2d` (`s=0.0`) | 6.823 | 3.060 | **0.4484** (**44.8% noise leakage, 6.8x blowup — FAIL**) |
| `unified_v12`/`v13` BF16 `quintic_orthogonalize_2d` (`s=0.0061`) | 0.939 | 3.99e-6 | 4.25e-6 |
| Exact FP64 polar reference | 1.000 | 8.11e-9 | 8.11e-9 |

---

## Suite 2 — Law 4.4 (P1 5-Regime Scale Invariance, Rail Escape, Spike Recovery & Settling)

All five presets' generator optimizers evaluated at matched `lr=0.02` (`lr=0.08` for `1e-12` tanh rail escape).

| Preset | 24-Decade `max_diff` (`1e-12..1e12`) | `u(1e-40)` Move | Rail `1e-6` / `1.0` / `1e6` Final RMS | `1e4x` Spike `c=1e-6` Final Err (`conv?`) | `1/t` Decay `s2000/s1` | Sign-Flip Chatter `tail/s1` | Bilinear Rot `r_end/r0` |
|---|---:|---:|---|---:|---:|---:|---:|
| **`axiom`** | **1.61e-6** | **0.0566** | **1.95e-5 / 1.95e-5 / 1.95e-5** (esc `163`) | **1.95e-5** (**True**) | **1.000** | **0.00097** | 1.184 |
| `unified_v12` | 0.3257 | 0.0000 | 0.0 / 0.0198 / 0.0150 (esc `182/180/170`) | 3.0243 (False) | 0.0012 | 1.00000 | 1.151 |
| `unified_v13` | 0.3257 | 0.0000 | 0.0 / 0.0198 / 0.0150 (esc `182/180/170`) | 3.0243 (False) | 0.0012 | 1.00000 | 1.151 |
| `minimal` | 0.1764 | 0.0000 | 7.00 / 7.00 / 0.0 (trapped `1e-6, 1.0`) | 2.9290 (False) | 0.0204 | 1.00000 | **0.634** |
| `rhbi` | 0.6061 | 0.0000 | 7.00 / 7.00 / 6.85 (trapped all 3) | 3.5425 (False) | 0.0005 | 1.00000 | 1.117 |

**Takeaway:** Axiom's IEEE-754 mantissa-exponent normalization (`frexp`/`ldexp`) eliminates both the `min(s_base)` scale-breaking cap and subnormal underflow (`max_diff = 1.61e-6` across 24 decades; full `0.0566` step even at `c = 1e-40` where all four baselines freeze at `0.0`). In `axiom-v4`, the `V4E` equilibrated step-cosine & Tikhonov controller (deadband-free `log2_damp in [-10, 0]` decaying at `rho_n < -0.75` and releasing rapidly at `rho_n > -0.65`, `dec=0.02`, zero permanent `s_eq` floor inside 2D NS) achieves scale-identical `1e-12` rail escape at step `163` (`final_rms = 1.95e-5` identically across 12 decades `1e-6..1e6`), complete `10^4x` spike immunity (`post/pre = 1.0000`, `final_err = 1.95e-5` at both `c=1e-6` and `c=1.0`), `1.000` (`0.999995`) non-freezing under coherent `1/t` decay, and `0.00097` (`2^-10`) period-2 chatter damping. On unregularized alternating bilinear rotation (600 steps, `4x4` matrix), exact all-2D polar NS has `r_end/r0 = 1.184`, matching `unified_v12`/`v13` (`1.151`) and `rhbi` (`1.117`) because full-rank polar updates take Frobenius step $\sqrt{m}\,\eta$ (whereas `minimal`'s `0.634` contraction comes from scale-breaking fixed-$\epsilon$ Adam damping).

---

## Suite 3 — Law 4.1 & 4.3 (P4 Verified Iterative Stratified `p95` Pushforward Trust)

Tests execute each recipe's actual generator hook and optimizer trust path with `d_max = 0.20` (`push_res = 0.96875`, matched `ema_data_scale = sqrt(2)`).

| Preset | 3A-1 Saturating `tanh` (`p95/d_max`, bt) | 3A-2 Superlinear `cubic` (`alpha`, `p95/d_max`) | 3A-3 `16L` `tanh` MLP (`p95/d_max`, bt) | 3B `coord_scale` Ratio (`1e-4` / `1e-3` / `1.0` / `1e4`) | 3C `1.6%` Rare Tail (`captured`, `rms/d_max`, `max`) |
|---|---|---|---|---|---|
| **`axiom`** | **0.54x (`4` bt)** (**PASS**) | **1.00, 0.0025x** (**PASS**) | **0.54x (`2` bt)** (**PASS**) | **1.000 / 1.000 / 1.000 / 1.000** (**PASS**) | **True, 0.25x, 0.400** (**PASS**) |
| `unified_v12` | 5.00x (`0` bt) (FAIL) | 16.00, **10.24x** (FAIL) | 1.15x (`0` bt) (FAIL) | **500.0 / 50.0** / 1.000 / 1.000 (FAIL) | False, 2.65x, 4.243 (FAIL) |
| `unified_v13` | 5.00x (`0` bt) (FAIL) | 16.00, **10.24x** (FAIL) | 1.15x (`0` bt) (FAIL) | **500.0 / 50.0** / 1.000 / 1.000 (FAIL) | False, 2.65x, 4.243 (FAIL) |
| `minimal` | 5.00x (`0` bt) (FAIL) | 69.61, **300.96x** (FAIL) | 1.01x (`0` bt) (FAIL) | **500.0 / 50.0** / 1.000 / 1.000 (FAIL) | False, 1.20x, 1.915 (FAIL) |
| `rhbi` | 5.00x (`0` bt) (FAIL) | 1.00, 0.0025x | 4.16x (`0` bt) (FAIL) | N/A (no probe) | False, 2.65x, 4.243 (FAIL) |

**Takeaway:**
- **Zero post-rescale violations:** Axiom's 4-step verified backtracking achieves `p95/d_max = 0.54x <= 1.0` on both the saturating `tanh` cliff (`v12`/`v13`/`minimal`/`rhbi` violate by `5.00x`) and the 16-layer `tanh` MLP (`v12`/`v13` violate by `1.15x`, `minimal` by `1.01x`, `rhbi` by `4.16x`). Removing upward `delta_g_min` amplification prevents `v12`/`v13`'s `10.24x` (`alpha=16.0`) and `minimal`'s `300.96x` (`alpha=69.61`) superlinear cubic overshoots.
- **Exact scale homogeneity & shock immunity:** Removing the hardcoded `max(0.05, ...)` floor shared by `v12`/`v13` and `minimal` yields `1.000000x` `coord_scale` fidelity across `[1e-4, 1e4]` (vs `50x`–`500x` distortion in `v12`/`v13`/`minimal`), while `axiom-v4`'s log-space clamped `ema_data_scale` update bounds a single `10^4x` shock batch to `<= 1.035x` drift.
- **Stratified `p95` tail capture:** On `B=128` with a 2-sample (`1.56%`) rare tail cluster at indices `126..127`, Axiom's 32-sample stratified hook (`16` uniform + `16` top-norm) captures both tail points (`2/32 = 6.25% > 5%`), triggers `1` backtrack (`alpha = 0.094`), cuts `rms/d_max` from `2.65x` to `0.25x`, and shrinks worst-sample jump by **10.6x** (`4.243 -> 0.400`), whereas `v12`/`v13`'s prefix `[:16]` probe misses the tail completely (`alpha = 1.0`, `max = 4.243`).

---

## Suite 4 — Law 2 (P6 Zero-Jensen Linear-R1, Dimension-Independent $\mu_D > 0$, & AD Safety)

- **4A (Below-$\kappa$ Unbiasedness):** `d=24` MLP calibrated to $\|\nabla_x D\|_2 = 0.65 < \kappa = 1.0$ with zero mean logit; live `D(xr), D(xf)` run before `penalty(...)` to match real `GANTrainer` execution; averaged over `128` MC draws vs exact RoR target gradient.
- **4B (Dimension-Independent $\mu_D > 0$):** `dim=2` MLP vs `dim=196,608` BN-free `256x256x3` Conv2d critic, both calibrated to $\|\nabla_x D\|_2 = 0.25 \ll \kappa = 1.0$ ($s_{\text{true}} \approx 0.062$) and zero mean logit; `eff_coeff = penalty / s_true`.
- **4C (Architectural AD Safety):** `EmulatedLinear` (`mxfp8`, `nvfp4`), custom `_NoJvpCritic` (no `jvp`), and `DCGANCritic(64, 32, 128)` under CUDA `bf16` autocast (plus `InstanceNorm2d` & `GroupNorm+LayerNorm` BF16 `fwAD` coverage in `tests/test_axiom_critic_opt.py`).

| Preset | 4A Loss `MC/Exact` | 4A Grad Norm `MC/Exact` | 4A Grad Cosine | 4B `eff_2d` | 4B `eff_r256` | 4B `r256 / 2d` Ratio | 4C Compat (`mxfp8 / nvfp4 / no_jvp / bf16`) |
|---|---:|---:|---:|---:|---:|---:|---|
| **`axiom`** | **1.07x** | **1.09x** | **0.9896** | **2.50e-3** | **2.65e-3** | **1.06x** (**PASS**) | **PASS / PASS / PASS / PASS** |
| `unified_v12` | **1423.29x** | **185.48x** | 0.9933 | 5.00e-3 | 1.80e-5 | **3.60e-3** (`278x` dilute) | PASS / PASS / PASS / PASS |
| `unified_v13` | **1423.29x** | **185.48x** | 0.9933 | 5.00e-3 | 1.80e-5 | **3.60e-3** (`278x` dilute) | PASS / PASS / PASS / PASS |
| `minimal` | **0.65x** | 0.95x | **0.5363** | 4.50e-1 | 8.78e+2 | **1953x** blowup | PASS / PASS / PASS / PASS |
| `rhbi` | **0.004x** | **0.007x** | 0.9989 | 4.96e-13 | 3.51e-15 | 7.09e-3 (vanishes) | PASS / PASS / PASS / PASS |

**Takeaway:** Axiom's detached-EMA linear-in-$\hat{s}$ R1+R2 stochastic `fwAD` path eliminates BOTH `v12`/`v13`'s **1423x** convex-EBI Jensen false-triggering below $\kappa$ AND `minimal`'s **-35%** concave-BI negative bias / `0.536` directional distortion, while holding exact dimension-independent normal dissipation $\mu_D = \lambda_{\text{floor}}/2 = 0.0025$ across a **98,304x** increase in input dimension (`dim=2 -> 196,608`, observed ratio `1.06x` vs `v12`/`v13`'s `278x` dilution and `rhbi`'s `3.5e-15` vanishing floor). All four architectural AD boundaries (`mxfp8`, `nvfp4`, `no_jvp`, `DCGANCritic bf16`) pass cleanly with finite values and gradients.

---

## Suite 5 — Law 1 (P7 Mass Conservation, Exact LTV, $B_f < K$ Quotas & Host Syncs)

### 5A–C. Prototype Atlas Invariants

| Preset | 5A Shock `abs(sum(w_fake_ema)-1)` | 5A `abs(sum(proto_w)-1)` | 5B Gersho Merge LTV | 5B Split LTV | 5B `10x` False Merge | 5B Post-Coal `abs(mean(psi))` | 5C $B_f < K$ Starved Allocated (`of 36`) |
|---|---:|---:|---:|---:|---|---:|---:|
| **`axiom`** | **5.96e-8** (**PASS**) | **1.19e-7** | **1.0000** (**PASS**) | **1.0000** (**PASS**) | **False** | **1.99e-8** | **36 / 36** (`0` perm err) |
| `unified_v12` | 0.1187 (FAIL) | 1.79e-7 | 0.2667 (-73.3% FAIL) | 0.8100 (-19.0% FAIL) | False | — | 0 / 36 (legacy `36` evicted) |
| `unified_v13` | 0.1187 (FAIL) | 1.79e-7 | 0.2667 (-73.3% FAIL) | 0.8100 (-19.0% FAIL) | False | — | 0 / 36 (legacy `36` evicted) |
| `minimal` | 0.1187 (FAIL) | 1.79e-7 | N/A | N/A | **True** (FAIL) | **0.800** (FAIL) | N/A |
| `rhbi` | N/A (no atlas) | N/A | N/A | N/A | N/A | N/A | N/A |

### 5D. Real `GANTrainer` GPU-CPU Host Syncs & Step Latency (`CUDA_VISIBLE_DEVICES=0`)

Measured over 10 steady-state steps after warmup on both `grid100` MLP (`bsz=256`, `1024` MoG particles) and `64x64` DCGAN Conv2d (`bsz=16`, `64` particles).

| Preset | Conv2d `64x64` `ms/step` | Conv2d `syncs/step` (`item/tolist/bool/uniq`) | MLP `grid100` `ms/step` | MLP `syncs/step` (`item/tolist/bool/uniq`) |
|---|---:|---|---:|---|
| **`axiom`** | **21.48** (**#1 fastest**) | **4.0** (**#1 fewest** `2.0 / 2.0 / 0.0 / 0.0`) | 44.37 | 35.9 (`23.9 / 6.0 / 5.0 / 1.0`) |
| `unified_v12` | 22.88 | 5.0 (`0.0 / 3.0 / 2.0 / 0.0`) | 32.82 | 10.0 (`2.0 / 5.0 / 2.0 / 1.0`) |
| `unified_v13` | 21.79 | 5.0 (`0.0 / 3.0 / 2.0 / 0.0`) | 32.73 | 10.0 (`2.0 / 5.0 / 2.0 / 1.0`) |
| `minimal` | 27.96 | 6.0 (`3.0 / 0.0 / 2.0 / 1.0`) | 57.29 | 28.0 (`19.0 / 0.0 / 7.0 / 2.0`) |
| `rhbi` | 91.99 | 75.0 (`0.0 / 0.0 / 74.0 / 1.0`) | **25.45** | **4.0** (`0.0 / 0.0 / 3.0 / 1.0`) |

---

## Honest Findings & Concrete Optimization Follow-Ups

1. **All 9 Structural Four-Laws Controls Pass on `axiom-v4` (`59a860fa0`, vs `0/9` on every legacy family):** Every mathematical defect identified in the Four-Laws Audit — role-asymmetric critic NS omission (`hq=0.9417`, `mass_tv=0.0229` on 16L-D AND #1 mode coverage `23/100` on early 500-step 16L-G), BF16 rank-1 NS leakage (`1.29e-5` vs `44.8%`), `min(s_base)` scale breaking & subnormal freeze (`1.61e-6` across 24 decades; full `0.0566` move at `1e-40`), monotonic `peak_rms` spike freeze (`final_err=1.95e-5`), constant-amplitude chatter (`0.00097`), one-shot secant overshoot & upward cubic amplification (`0.54x / 0.00x / 0.54x`), `coord_scale` `0.05` floor (`1.000x` across `[1e-4, 1e4]`), unstratified tail blindness (`10.6x` jump reduction), convex-EBI Jensen false-triggering (`1.07x` vs `1423x`), high-res $\mu_D$ dilution (`1.06x` vs `278x` dilution), shock `w_fake_ema` mass leak (`5.96e-8` vs `0.119`), Gersho merge/split LTV loss (`1.0000 / 1.0000`), and $B_f < K$ index eviction (`36/36` vs `0/36`) — passes cleanly on `axiom-v4` (`2,399 / 2,500` LOC).
2. **#1 Conv2d Throughput & Host Syncs (`21.48 ms/step`, `4.0 syncs/step`):** On Conv2d architectures, `axiom-v4`'s 3-kernel `_run_foreach_fast` path plus Weyl-phase `b_reg = B // 4` interleaved `fwAD` leads all five families in BOTH latency (`21.48 ms` vs `v13` `21.79 ms`, `v12` `22.88 ms`, `minimal` `27.96 ms`, `rhbi` `91.99 ms`) and GPU-CPU host syncs (`4.0` vs `v12`/`v13` `5.0`, `minimal` `6.0`, `rhbi` `75.0`), while keeping `_capture_ring` strictly zero-allocation (`0` entries) on high-res inputs (`> 4096` elements/sample).
3. **MLP Host-Sync Profile (`35.9` syncs/step on `grid100`):** Remaining `23.9 .item()` calls/step on low-D `grid100` MLP come primarily from Law-1 pullback quantizer/chart scalar checks in `prior.py` (`v12`/`v13` run `10.0` syncs / `32.8 ms/step`), while `axiom-v4`'s `torch._foreach_norm` / `_foreach_add_` + `pre_normalized=True` NS + CPU small-atlas optimizations still outperform `minimal` (`57.29 ms/step`) by 23% and keep `AE-DDGAN` / `five_word` well inside Forge wall budgets (see [Axiom Forge Study](axiom-study.md)).
4. **Transparent Trade-Offs (`1B` Early 500-Step Snapshot & Bilinear Rotation):** On the 500-step early snapshot of `grid100_mlp16` (1B), `axiom-v4` leads all baselines (`23/100` vs `v12` `5/100`, `v13` `7/100`) while ramping more conservatively than `axiom-v3` (`100/100`) due to `V4E`'s faster `dec=0.02` & non-2D `s_eq_base=0.05` equilibrium settling. On unregularized alternating bilinear rotation (600 steps, `4x4` matrix), exact all-2D polar NS has `r_end/r0 = 1.184` (comparable to `v12`/`v13` `1.151` and `rhbi` `1.117`), while real minimax games are stabilized by Law 2 normal dissipation ($\mu_D > 0$).
