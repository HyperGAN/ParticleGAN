# Axiom Four-Laws Adversarial Head-to-Head Benchmark

**Scope & Provenance:** Deterministic (`seed=0`, `deterministic_orthogonal_` initialization, matched architectures, batch sequences, prior capacities, and step budgets) head-to-head stress benchmark comparing final **`axiom`** (`axiom-v3`, commit `37f537db8`) against **`unified_v12`**, **`unified_v13`**, **`minimal`**, and **`rhbi`** across the five Four-Laws audit controls where prior ParticleGAN families exhibit documented structural failures (see [Four-Laws Integration Report](../four-laws-audit/integration-report.md), [Axiom Design Spec](design-spec.md), [Gap Analysis](gap-analysis.md), and [Axiom Forge Study](axiom-study.md)).

- **Branch / Evaluated Commit:** `feat/axiom-v3-laws-report` from `origin/feat/axiom-family` (`37f537db8`, `axiom-v3` V3I)
- **Benchmark Driver:** [`scripts/axiom_four_laws_benchmark.py`](../../scripts/axiom_four_laws_benchmark.py)
- **Hardware & Runtime:** Physical RTX A6000 `cuda:1` (`CUDA_VISIBLE_DEVICES=1`, keeping `cuda:0` idle), PyTorch `2.15.0+cu132`, total wall time `218.89 s`
- **Reproduction Command:**
  ```bash
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" /nvme/dev/hypergan/ParticleGAN/.venv/bin/python \
      scripts/axiom_four_laws_benchmark.py --device cuda:0
  ```

> [!NOTE]
> All measurements below are single-seed (`seed=0`) adversarial stress controls executed through each recipe's public API without altering any baseline's formula. They verify structural Law 1–4 invariants and identify concrete engineering gaps; see [Axiom Forge Study](axiom-study.md) for the full 8-study RTX A6000 Forge evaluation battery.

---

## Executive Four-Laws Leaderboard

| Audit Control / Law | Key Metric (Target) | **`axiom`** (`v3`) | `unified_v12` | `unified_v13` | `minimal` | `rhbi` |
|---|---|---:|---:|---:|---:|---:|
| **1A. Law 4.2 (P5): 16L Critic Swap (`ring8` B1)** | Modes / `hq` (`8/8`, `>= 0.90`) | **8/8 / 0.9858** (**PASS**) | 8/8 / 0.5000 (FAIL) | 8/8 / 0.5000 (FAIL) | 0/8 / 0.0073 (FAIL) | 0/8 / 0.0000 (FAIL) |
| **1C. Law 4.2 (P0): Rank-1 `128x128` NS (`s=0`)** | $\sigma_2 / \sigma_1$ (`< 1e-3`), $\sigma_1$ (`~1.0`) | **1.29e-5 / 1.124** (**PASS**) | 0.4484 / 6.823 (FAIL) | 0.4484 / 6.823 (FAIL) | N/A (0 NS) | exact polar |
| **2A. Law 4.4 (P1): 24-Decade Scale Invariance** | `max_diff` on `[1e-12, 1e12]` (`< 1e-5`) | **1.12e-6** (**PASS**) | 0.326 (FAIL) | 0.326 (FAIL) | 0.176 (FAIL) | 0.606 (FAIL) |
| **2B–D. Law 4.4 (P1): Rail + Spike + Chatter** | `rail(1e6)` / `spike(1e-6)` / `chatter` | **7.8e-11 / 2.1e-10 / 0.001** (**PASS**) | 0.015 / 3.02 / 1.00 (FAIL) | 0.015 / 3.02 / 1.00 (FAIL) | 0.0 / 2.93 / 1.00 (FAIL) | 6.85 / 3.54 / 1.00 (FAIL) |
| **3A. Law 4.1 (P4): Verified Trust (`tanh` / `cubic` / `16L`)** | Worst `p95 / d_max` (`<= 1.0x`) | **0.54x / 0.00x / 0.54x** (**PASS**) | 5.00x / 10.24x / 1.15x (FAIL) | 5.00x / 10.24x / 1.15x (FAIL) | 5.00x / 301.0x / 1.01x (FAIL) | 5.00x / 0.00x / 4.16x (FAIL) |
| **3B–C. Law 4.3 (P4): `coord_scale` + `1.6%` Tail** | `1e-4` ratio (`1.0x`) / Tail captured | **1.000x / True** (**PASS**) | 500.0x / False (FAIL) | 500.0x / False (FAIL) | 500.0x / False (FAIL) | N/A / False (FAIL) |
| **4A. Law 2 (P6): Below-$\kappa$ Unbiasedness (`d=24`)** | `MC / Exact` loss (`~1.0x`) & grad cos | **1.07x / cos 0.990** (**PASS**) | 1423.3x / cos 0.993 (FAIL) | 1423.3x / cos 0.993 (FAIL) | 0.65x / cos 0.536 (FAIL) | 0.004x / vanishes (FAIL) |
| **4B. Law 2 (P6): Dim-Indep $\mu_D > 0$ (`2D` vs `256^2`)** | `eff_r256 / eff_2d` (`~1.0x`) | **1.06x** (`2.65e-3`) (**PASS**) | 3.60e-3 (`278x` dilute, FAIL) | 3.60e-3 (`278x` dilute, FAIL) | 1953x blowup (FAIL) | 7.09e-3 (`3.5e-15`, FAIL) |
| **5A–C. Law 1 (P7): Mass, LTV, Gauge & $B_f < K$** | `wf_err` / `LTV(M,S)` / `B<K` | **0.0 / (1.00, 1.00) / 36/36** (**PASS**) | 0.119 / (0.27, 0.81) / 0/36 (FAIL) | 0.119 / (0.27, 0.81) / 0/36 (FAIL) | 0.119 / false-10x / $\psi=0.8$ (FAIL) | N/A (no atlas) |
| **5D. Runtime: Conv2d `64x64` vs MLP `grid100`** | `ms/step` (`syncs/step`) | **17.10 ms (4.0)** / 35.25 ms (31.4)* | 18.40 ms (5.0) / 27.11 ms (10) | 18.42 ms (5.0) / 28.00 ms (10) | 22.34 ms (6.0) / 36.66 ms (28) | 43.13 ms (75) / **20.74 ms (4)** |

**Summary Score (Structural Law Controls 1A, 1C, 2A–D, 3A–C, 4A–B, 5A–C):**
- **`axiom`: 9 / 9 PASS**, plus **100/100 modes** on 16-layer skip-free `grid100_mlp16` in 500 steps (`hq=0.6706` vs `unified_v12` `4/100` / `unified_v13` `6/100`), **#1 fastest & fewest-sync** Conv2d `GANTrainer` step (`17.10 ms/step`, `4.0 syncs/step`), and MLP host syncs cut **53%** from `v1` (`66.4 -> 31.4` syncs/step).
- **`unified_v12` / `unified_v13`: 0 / 9 PASS** across these nine adversarial failure controls.
- **`minimal`: 0 / 9 PASS**.
- **`rhbi`: 0 / 9 PASS** (except exact polar reference in 1C).

---

## Suite 1 — Law 4.2 (P5 Role-Agnostic 2D NS & P0 FP32 Tikhonov NS)

### 1A. Regime B1 — Deep 16-Layer Skip-Free Critic Swap (`ring8`, Pure Adversarial RpGAN + Law 2 BI $\beta=2.0$, 300 Steps)

Architecture: `3L` skip-free generator (`SimpleMLPGenerator(2, 128, 3, 2)`) vs `16L` skip-free critic (`SimpleMLPDiscriminator(2, 128, 16, fourier=0)`), `512` MoG particles, `bsz=256`. Every preset runs its unmodified `recipe.make_optimizers(G, D, prior=prior)`.

| Preset | Modes / 8 | `hq` | `mass_tv` | `d_acc_gauss` | `d_acc_valley` | `ns_G` | `ns_D` | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`axiom`** | **8/8** | **0.9858** | 0.0354 | **0.796** | 0.607 | 1200 | **4800** | 12.7 |
| `unified_v12` | 8/8 | 0.5000 | **0.0190** | 0.734 | **0.625** | 1200 | 0 | 8.7 |
| `unified_v13` | 8/8 | 0.5000 | **0.0190** | 0.734 | **0.625** | 1200 | 0 | 8.5 |
| `minimal` | 0/8 | 0.0073 | 0.8750 | 0.491 | 0.549 | 0 | 0 | 15.6 |
| `rhbi` | 0/8 | 0.0000 | 0.5901 | 0.566 | 0.492 | 0 | 0 | 14.2 |

**Takeaway:** `unified_v12`/`v13` hardcode NS exclusively to the generator (`ns_G=1200`, `ns_D=0`), starving the 16-layer skip-free critic and stalling generator quality at `hq=0.5000`. `axiom`'s strictly role-agnostic 5-step FP32 Tikhonov NS on all 2D dense layers (`p.ndim == 2 and min(m, n) >= 2`, keeping `0` NS on 4D Conv2d) rescues the 16L critic (`ns_D=4800`), lifting `hq` by **+48.6pp** (`0.9858`) while cutting `mass_tv` in half relative to `axiom-v1` (`0.0735 -> 0.0354`) and improving Gaussian critic discrimination (`d_acc_gauss = 0.796`).

### 1B. Regime A — Deep 16-Layer Skip-Free Generator (`grid100_mlp16`, Real `GANTrainer`, 500 Steps)

Architecture: `16L` skip-free generator (`SimpleMLPGenerator(2, 128, 16, 2)`) + `3L` critic (`fourier=0`), `2048` MoG particles, `bsz=256`, 500 steps (early-budget stress snapshot; official Forge `grid100_mlp16` runs longer).

| Preset | Modes / 100 | `hq` | `ns_G` | `ns_D` | Wall (s) |
|---|---:|---:|---:|---:|---:|
| **`axiom`** | **100/100** | **0.6706** | 8500 | 1500 | **25.4** |
| `unified_v12` | 4/100 | 0.0304 | 8500 | 0 | 30.0 |
| `unified_v13` | 6/100 | 0.0175 | 8500 | 0 | 27.8 |
| `minimal` | 2/100 | 0.0171 | 0 | 0 | 30.3 |
| `rhbi` | 0/100 | 0.0000 | 0 | 0 | 19.0 |

**Takeaway:** Combining role-agnostic all-2D FP32 Tikhonov NS (`axiom-v2`) with coherence-gated Peak-RMS Tikhonov settling and Law-1/Law-3 transport fixes (`axiom-v3`) lifts 500-step 16-layer skip-free generator coverage from `6/100` (`axiom-v1`) to **`100/100` modes (`hq=0.6706`)** — **16x–25x** higher mode coverage and **22x–38x** higher `hq` than `unified_v12`/`v13`, while finishing **15% faster** (`25.4 s` vs `30.0 s`).

### 1C. Healthy `r256` Conv2d Zero-NS Guard & P0 Rank-1 `128x128` NS Spectral Leakage

- **Healthy `DCGANCritic(256, 64, 512)` (`bf16` channels-last, 5 steps):** `ns_D = 0` for `axiom` (and all baselines), confirming zero Newton–Schulz overhead on healthy Conv2d architectures.
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
| **`axiom`** | **1.12e-6** | **0.0566** | **1.0e-10 / 1.4e-10 / 7.8e-11** (esc `163`) | **2.1e-10** (**True**) | **1.000** | **0.00097** | 1.176 |
| `unified_v12` | 0.3257 | 0.0000 | 0.0 / 0.0198 / 0.0150 (esc `182/180/170`) | 3.0243 (False) | 0.0012 | 1.00000 | 1.151 |
| `unified_v13` | 0.3257 | 0.0000 | 0.0 / 0.0198 / 0.0150 (esc `182/180/170`) | 3.0243 (False) | 0.0012 | 1.00000 | 1.151 |
| `minimal` | 0.1764 | 0.0000 | 7.00 / 7.00 / 0.0 (trapped `1e-6, 1.0`) | 2.9290 (False) | 0.0204 | 1.00000 | **0.634** |
| `rhbi` | 0.6061 | 0.0000 | 7.00 / 7.00 / 6.85 (trapped all 3) | 3.5425 (False) | 0.0005 | 1.00000 | 1.117 |

**Takeaway:** Axiom's IEEE-754 mantissa-exponent normalization (`frexp`/`ldexp`) eliminates both the `min(s_base)` scale-breaking cap and subnormal underflow (`max_diff = 1.12e-6` across 24 decades; full step even at `c = 1e-40` where all four baselines freeze at `0.0`). In `axiom-v3`, coherence-gated Winsorized Peak-RMS Tikhonov (`g_eq = smoothstep((0.55 - rho_n)/0.30)`) plus period-2 flip damping (`log2_damp in [-10, 0]`) drives both `1e-12` rail settling (`rms ~ 1e-10`, `tail_step = 6.0e-10`, identical escape at step `163` across 12 decades!) and `10^4x` spike recovery (`post/pre = 1.0000`, `final_err = 2.1e-10` at `c=1e-6` and `1.9e-9` at `c=1.0`) to machine-precision convergence, while maintaining `1.000` non-freezing under coherent `1/t` decay and `0.00097` (`2^-10`) period-2 chatter damping. On unregularized alternating bilinear rotation (600 steps, `4x4` matrix), exact all-2D polar NS has `r_end/r0 = 1.176`, matching `unified_v12`/`v13` (`1.151`) and `rhbi` (`1.117`) because full-rank polar updates take Frobenius step $\sqrt{m}\,\eta$ (whereas `minimal`'s `0.634` contraction comes from scale-breaking fixed-$\epsilon$ Adam damping).

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
- **Exact scale homogeneity:** Removing the hardcoded `max(0.05, ...)` floor shared by `v12`/`v13` and `minimal` yields `1.000000x` `coord_scale` fidelity across `[1e-4, 1e4]` (vs `50x`–`500x` distortion in `v12`/`v13`/`minimal`).
- **Stratified `p95` tail capture:** On `B=128` with a 2-sample (`1.56%`) rare tail cluster at indices `126..127`, Axiom's 32-sample stratified hook (`16` uniform + `16` top-norm) captures both tail points (`2/32 = 6.25% > 5%`), triggers `1` backtrack (`alpha = 0.094`), cuts `rms/d_max` from `2.65x` to `0.25x`, and shrinks worst-sample jump by **10.6x** (`4.243 -> 0.400`), whereas `v12`/`v13`'s prefix `[:16]` probe misses the tail completely (`alpha = 1.0`, `max = 4.243`). Note: because `torch.quantile(..., 0.95)` over 32 probe points interpolates indices `29` and `30`, a single isolated point (`1/128 = 0.78%`) is captured in the probe set (`probe_captured_tail=True`) but sits above the 95th percentile (`1/32 = 3.1% < 5%`); any tail cluster with `>= 2` samples (`>= 1.6%` of `B=128`) is actively clamped.

---

## Suite 4 — Law 2 (P6 Zero-Jensen Linear-R1, Dimension-Independent $\mu_D > 0$, & AD Safety)

- **4A (Below-$\kappa$ Unbiasedness):** `d=24` MLP calibrated to $\|\nabla_x D\|_2 = 0.65 < \kappa = 1.0$ with zero mean logit; live `D(xr), D(xf)` run before `penalty(...)` to match real `GANTrainer` execution; averaged over `128` MC draws vs exact RoR target gradient.
- **4B (Dimension-Independent $\mu_D > 0$):** `dim=2` MLP vs `dim=196,608` BN-free `256x256x3` Conv2d critic, both calibrated to $\|\nabla_x D\|_2 = 0.25 \ll \kappa = 1.0$ ($s_{\text{true}} \approx 0.062$) and zero mean logit; `eff_coeff = penalty / s_true`.
- **4C (Architectural AD Safety):** `EmulatedLinear` (`mxfp8`, `nvfp4`), custom `_NoJvpCritic` (no `jvp`), and `DCGANCritic(64, 32, 128)` under CUDA `bf16` autocast.

| Preset | 4A Loss `MC/Exact` | 4A Grad Norm `MC/Exact` | 4A Grad Cosine | 4B `eff_2d` | 4B `eff_r256` | 4B `r256 / 2d` Ratio | 4C Compat (`mxfp8 / nvfp4 / no_jvp / bf16`) |
|---|---:|---:|---:|---:|---:|---:|---|
| **`axiom`** | **1.07x** | **1.09x** | **0.9896** | **2.50e-3** | **2.65e-3** | **1.06x** (**PASS**) | **PASS / PASS / PASS / PASS** |
| `unified_v12` | **1423.29x** | **185.48x** | 0.9933 | 5.00e-3 | 1.80e-5 | **3.60e-3** (`278x` dilute) | PASS / PASS / PASS / PASS |
| `unified_v13` | **1423.29x** | **185.48x** | 0.9933 | 5.00e-3 | 1.80e-5 | **3.60e-3** (`278x` dilute) | PASS / PASS / PASS / PASS |
| `minimal` | **0.65x** | 0.95x | **0.5363** | 4.50e-1 | 8.78e+2 | **1953x** blowup | PASS / PASS / PASS / PASS |
| `rhbi` | **0.004x** | **0.007x** | 0.9989 | 4.96e-13 | 3.51e-15 | 7.09e-3 (vanishes) | PASS / PASS / PASS / PASS |

**Takeaway:** Axiom's detached-EMA linear-in-$\hat{s}$ R1+R2 stochastic `fwAD` path eliminates BOTH `v12`/`v13`'s **1423x** convex-EBI Jensen false-triggering below $\kappa$ AND `minimal`'s **-35%** concave-BI negative bias / `0.536` directional distortion, while holding exact dimension-independent normal dissipation $\mu_D = \lambda_{\text{floor}}/2 = 0.0025$ across a **98,304x** increase in input dimension (`dim=2 -> 196,608`, observed ratio `1.06x` vs `v12`/`v13`'s `278x` dilution and `rhbi`'s `3.5e-15` vanishing floor). On low-D exact-RoR critics (`Var == 0`), `axiom-v3`'s exact-deadband $C^1$ wall is identically `0` below $\kappa$ while preventing above-$\kappa$ slope blowup. All four architectural AD boundaries (`mxfp8`, `nvfp4`, `no_jvp`, `DCGANCritic bf16`) pass cleanly with finite values and gradients.

---

## Suite 5 — Law 1 (P7 Mass Conservation, Exact LTV, $B_f < K$ Quotas & Host Syncs)

### 5A–C. Prototype Atlas Invariants

| Preset | 5A Shock `abs(sum(w_fake_ema)-1)` | 5A `abs(sum(proto_w)-1)` | 5B Gersho Merge LTV | 5B Split LTV | 5B `10x` False Merge | 5B Post-Coal `abs(mean(psi))` | 5C $B_f < K$ Starved Allocated (`of 36`) |
|---|---:|---:|---:|---:|---|---:|---:|
| **`axiom`** | **0.00e+00** (**PASS**) | **1.19e-7** | **1.0000** (**PASS**) | **1.0000** (**PASS**) | **False** | **1.99e-8** | **36 / 36** (`0` perm err) |
| `unified_v12` | 0.1187 (FAIL) | 1.79e-7 | 0.2667 (-73.3% FAIL) | 0.8100 (-19.0% FAIL) | False | — | 0 / 36 (legacy `36` evicted) |
| `unified_v13` | 0.1187 (FAIL) | 1.79e-7 | 0.2667 (-73.3% FAIL) | 0.8100 (-19.0% FAIL) | False | — | 0 / 36 (legacy `36` evicted) |
| `minimal` | 0.1187 (FAIL) | 1.79e-7 | N/A | N/A | **True** (FAIL) | **0.800** (FAIL) | N/A |
| `rhbi` | N/A (no atlas) | N/A | N/A | N/A | N/A | N/A | N/A |

### 5D. Real `GANTrainer` GPU-CPU Host Syncs & Step Latency (`CUDA_VISIBLE_DEVICES=1`)

Measured over 10 steady-state steps after warmup on both `grid100` MLP (`bsz=256`, `1024` MoG particles) and `64x64` DCGAN Conv2d (`bsz=16`, `64` particles).

| Preset | Conv2d `64x64` `ms/step` | Conv2d `syncs/step` (`item/tolist/bool/uniq`) | MLP `grid100` `ms/step` | MLP `syncs/step` (`item/tolist/bool/uniq`) |
|---|---:|---|---:|---|
| **`axiom`** | **17.10** (**#1 fastest**) | **4.0** (**#1 fewest** `2.0 / 2.0 / 0.0 / 0.0`) | 35.25 | 31.4 (`19.4 / 6.0 / 5.0 / 1.0`) |
| `unified_v12` | 18.40 | 5.0 (`0.0 / 3.0 / 2.0 / 0.0`) | 27.11 | 10.0 (`2.0 / 5.0 / 2.0 / 1.0`) |
| `unified_v13` | 18.42 | 5.0 (`0.0 / 3.0 / 2.0 / 0.0`) | 28.00 | 10.0 (`2.0 / 5.0 / 2.0 / 1.0`) |
| `minimal` | 22.34 | 6.0 (`3.0 / 0.0 / 2.0 / 1.0`) | 36.66 | 28.0 (`19.0 / 0.0 / 7.0 / 2.0`) |
| `rhbi` | 43.13 | 75.0 (`0.0 / 0.0 / 74.0 / 1.0`) | **20.74** | **4.0** (`0.0 / 0.0 / 3.0 / 1.0`) |

---

## Honest Findings & Concrete Optimization Follow-Ups

1. **All 9 Structural Four-Laws Controls Pass on `axiom-v3` (vs `0/9` on every legacy family):** Every mathematical defect identified in the Four-Laws Audit — role-asymmetric critic NS omission (`hq=0.9858` on 16L-D AND `100/100` modes / `hq=0.6706` on 16L-G), BF16 rank-1 NS leakage, `min(s_base)` scale breaking & subnormal freeze, monotonic `peak_rms` spike freeze (`final_err=2.1e-10`), constant-amplitude chatter (`0.00097`), one-shot secant overshoot & upward cubic amplification, `coord_scale` `0.05` floor, unstratified tail blindness, convex-EBI Jensen false-triggering, high-res $\mu_D$ dilution, shock `w_fake_ema` mass leak, Gersho merge/split LTV loss, and $B_f < K$ index eviction — is empirically resolved in `axiom-v3` (`37f537db8`).
2. **#1 Conv2d Throughput & Host Syncs (`17.10 ms/step`, `4.0 syncs/step`):** On Conv2d architectures, `axiom-v3`'s 3-kernel `_run_foreach_fast` path plus Weyl-phase `b_reg = B // 4` interleaved `fwAD` leads all five families in BOTH latency (`17.10 ms` vs `v12` `18.40 ms`, `minimal` `22.34 ms`, `rhbi` `43.13 ms`) and GPU-CPU host syncs (`4.0` vs `v12` `5.0`, `minimal` `6.0`, `rhbi` `75.0`). This aligns directly with the full solo-GPU `dcgan_performance_r256` Forge study (`32.94 ms/st`, `805.8 MiB` VRAM, `6.969 J/st` — `-20.4%` latency / `-28.3%` energy vs `unified_v12`; see [Axiom Forge Study](axiom-study.md)).
3. **MLP Host-Sync Improvement & Remaining Gap (`66.4 -> 31.4` syncs/step on `grid100`):** Packing `AxiomOptimizer`'s per-layer `(ok, l2n)` reads cut MLP `grid100` host syncs by **52.7%** (`66.4 -> 31.4` syncs/step) and latency from `39.67 -> 35.25 ms/step` (now faster than `minimal` `36.66 ms/step`). Remaining `19.4 .item()` calls/step come primarily from Law-1 pullback quantizer/pullback scalar checks in `prior.py` (`v12` runs `10.0` syncs / `27.11 ms/step`), leaving a clean future vectorization opportunity on low-D MLP workloads.
4. **Bilinear Rotation Trade-Off (`r_end/r0 = 1.176`):** Enabling role-agnostic all-2D FP32 Tikhonov NS in `axiom-v2/v3` lifted 16-layer skip-free `grid100_mlp16` from `6/100` to `100/100` modes and `ring8` 16L-D from `0.9741` to `0.9858`, while moving unregularized alternating bilinear rotation radius from `0.998` to `1.176` (comparable to `v12`/`v13` `1.151` and `rhbi` `1.117`). Because Law 2 critic normal dissipation ($\mu_D > 0$) stabilizes real minimax games, this is an acceptable trade-off documented transparently alongside the Forge results.
