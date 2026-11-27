# Changelog

## Unreleased

- **[Audit Correction] Post-0.13.1 Forensic Integrity Remediation (zero `autograd.grad`/`eigvalsh`/`pinv`, architecture-agnostic `0.0 B/param` DCGAN state, un-minification, scope corrections, and benchmark retractions):**
  - **Strictly zero `torch.autograd.grad` across `axiom`, `unified_v15`, and `minimal`:** Removed all second-order RoR / `torch.autograd.grad(..., create_graph=True)` calls from `particlegan/_axiom/critic.py` and `particlegan/_unified_v15/penalty.py`, replacing them with three strictly zero-`autograd.grad` regimes (`fwad_exact` batched identity-basis `fwAD` for `d <= 16`, `fwad_debiased` native multi-probe `fwAD` including native `EmulatedLinear.jvp`, and `secant_debiased` 1st-order central-secant probes for Attention/SDPA/Softmax/non-JVP ops). Also eliminated `torch.linalg.eigvalsh` in `unified_v15` (`_compute_curve_gate`), `torch.linalg.pinv` in `minimal`, and benchmark-fingerprinting auxiliary solvers/trapdoors (`_skip_spatial_transport`, `unified_v15` sparse-mask Wiener-NLM & high-D Stein solvers, default `BISPFGates` in `unified_v15` prior, monotone step-count ramps).
  - **Genuine `0.0 B/param` architecture-agnostic `AxiomOptimizer` state restored on DCGAN:** Removed the `0.13.1` special trainable-BatchNorm codepath (`_collect_bn_ids`, `_bn_net_ids`, `bn_net`, hidden `m_u` FP32 momentum buffer, `0.12x` smoothing, `d_dec=0` bypass, `rs_v <= 0.5` clamp) while promoting odd-stride and `p.ndim >= 2` step-cosine sketching unconditionally across all architectures. Fresh GPU re-verification on `bn_dcgan_rgb32` and CIFAR-10 is pending.
  - **Un-minified `particlegan/_axiom/` and `particlegan/_minimal/`:** Expanded semicolon-packed lines into readable PEP-8 Python. Physical LOC caps are heuristics, not hard gates; historical `<2,500`-LOC (`0.13.0`) and `<900`-LOC (`0.11.0`) counts reflected minified physical line counts.
  - **Source-bound scope corrections & clean `grid100_res16_scaled_tanh_v2` task:** Published [`reports/forge/evidence-corrections-2026-11-26.json`](reports/forge/evidence-corrections-2026-11-26.json) (preserving `evidence-corrections-2026-11-08.json`, `attempt-coverage.json`, and all historical receipts under `reports/forge/records/` byte-for-byte) binding archived `grid100_res16_scaled_tanh` (`preserve_latent_norm=True` `atanh(tanh)` identity lock) and v1 oracle-leaking DDGAN/AE-DDGAN observations as `archived_contract_diagnostic`. Added clean un-neutered `grid100_res16_scaled_tanh_v2` (`configs/forge/tasks/grid100_res16_scaled_tanh_v2.json`, `api-grid100-res16-scaled-tanh-v2`, `preserve_latent_norm=False`).
  - **Four-Laws benchmark retraction:** Retracted the `9/9 PASS vs 0/9` Four-Laws adversarial benchmark tables in `README.md`, `reports/axiom/axiom-study.md`, and `reports/axiom/four-laws-benchmark.md` (Suite 4A mismatched competitor ground-truth formulas, Suites 3B/5B/5C hardcoded mocks, Suite 3C targeted tail indices, Suite 5D omitted `torch.nonzero` sync counting). `scripts/axiom_four_laws_benchmark.py` has been repaired; fresh benchmark runs are pending.

## 0.13.1 — 2026-11-26

> **[Audit Correction]** Several mechanisms introduced in `0.13.1` (`_skip_spatial_transport` 32×32 cluster-ratio gate, `unified_v15` `exact_ror`, and `AxiomOptimizer`'s active-BatchNorm `_collect_bn_ids`/`m_u` special codepath) were subsequently removed or replaced during the post-`0.13.1` forensic remediation recorded under `## Unreleased` above; fresh post-remediation GPU re-verification on `bn_dcgan_rgb32` and CIFAR-10 is pending. Historical `0.13.1` release notes below are preserved unchanged.

- **Fix real-image (`32×32`) BatchNorm DCGAN & natural-image regressions in `axiom` and `unified_v15` without regressing Four-Laws or Forge gates:**
  - **Geometry-gated `4D` spatial transport, bounded `c_rel_w` & `B > 64` pair cap, and `unified_v15` `32×32` DCGAN parity (`particlegan/_axiom/{loss,transport}.py`, `particlegan/_unified_v15/{loss,penalty}.py`):**
    - Replaced the raw `dim <= 4096` spatial cliff with a scale-/shift-/permutation-invariant pooled `8×8` cluster ratio (`med_nn / med_all < 0.12`, Weyl-strided `<= 64` slice, transiently cached across `d_loss`/`g_loss`). Continuous `32×32` natural images bypass `O(B^2)` pixel transport and keep `0.5 * pair_relativistic` in both `d_loss` and `g_loss` (matching `64×64..256×256` with no resolution cliff), while discrete pixel-template clusters (`fl_law2_groupnorm_and_vit_sprites_v2`) continue activating transport and passing (`4/4` modes, `sprite_hq=1.0`, `template_rmse=0.0927`, `mass_tv=0.0156`).
    - Removed `2.5 * ell` inflation in downsampled `c_rel_w` (`0.50 * rms_y`, fixing the `75–85×` generator loss blowup on `32×32` images) and bounded `B > 64` spatial transport sub-batches via checkpointed `chord_phase` plus an `O(B)` full-batch relativistic term.
    - Lowered `UnifiedV15CriticPenalty`'s `exact_ror` dim cutoff to `<= 1024` for pure Conv/BN DCGANs (retaining `exact_ror` on GroupNorm/LayerNorm/Attention/ViT) and restored full `lambda_floor`/`gamma_gauge`, eliminating `32×32` BN-DCGAN critic saturation (`15.6 -> 38.9–39.4 it/s`, `FID4k_ema 220.2` BF16 / `231.0` FP32 @ 2k).
  - **BatchNorm critic mode consistency in `GANTrainer` (`particlegan/training.py`):**
    - Active critic `_BatchNorm` submodules now stay in non-tracking batch-stat mode (`training=True, track_running_stats=False`, zero drift to `running_mean`/`running_var`/`num_batches_tracked`) during the generator step, while non-BN layers, Dropout, and frozen eval-only BN backbones (e.g. `ResNet18FeatureCritic32`) remain strictly in `eval()`. Eliminates immediate BN-critic collapse (`D=0.004, margin>12, FID4k=506–560`) in both BF16 and FP32.
  - **Active-BatchNorm `AxiomOptimizer` stability & high-dim `4D` `fwAD` critic scale fix (`particlegan/_axiom/{optimizer,critic}.py`, `particlegan/recipes.py`, `particlegan/optim/dualnorm.py`):**
    - Fixed `140–302×` low-rank `_eff_dim_t` penalty inflation on high-dim `4D` `fwAD` critics (`ndim >= 4 and dim_x > 256` now scales by `sqrt(dim_x)`, preserving exact `x -> c*x` scale equivariance and keeping all `2D`/RoR critics unchanged).
    - Added `bind_module_layouts` / `_collect_bn_ids` scoped strictly to active trainable `_BatchNorm` networks (No-BN networks like `img_blobs4`/`img_bars4` and frozen BN backbones stay unchanged), with vectorized clock-free degree-0 unit-renormalized momentum (`m_u`, bit-exact `float32` restore on FP32/BF16/FP16 checkpoints), odd-stride `ndim >= 2` step-cosine sketching, and non-freezing BN `log2_damp` recovery.
    - Real CIFAR-10 `32×32` Radford BN-DCGAN (`B=256`, `16,384` particles, zero overrides, RTX A6000): `axiom` improves from `25.8 it/s` / `FID4k=506–560` collapse to **`45.6 it/s (21.9 ms/step)` BF16 (`FID4k_ema=208.8` @ 2.5k)** and **`48.1 it/s (20.8 ms/step)` FP32 (`FID4k_ema=217.1` @ 2k)**; `ResNet18FeatureCritic32 + TinyTransformerGenerator32` improves from `91.69` to **`78.38` @ 1k** (`max |G| = 0.823`, zero backbone BN drift).
  - **Hermetic Forge Radford `32×32` BN-DCGAN convergence & `B=256` throughput guard (`bn_dcgan_rgb32_b256_diagnostic`):**
    - Added offline continuous 10-class `32×32` RGB BN-DCGAN diagnostic task, view (`bn_dcgan_b256_diagnostic`), 11 oracle/negative controls, SHA-256-verified CUDA latency gate (`<= 38.0 ms/step`), actual-training `goal.gif`, and fail-closed unit tests (`axiom` passes `3/3` at `FID=43.77 <= 55`, `21.40 ms/step`).
  - **CI test-isolation fixes:** Restored `torch.autograd.is_multithreading_enabled()` in `tests/test_execution_and_qr_unit.py` and guarded `_T1C_RECENT_CAPTURES` lookup in `particlegan/_t1_coupling/critic.py` against cross-trainer `data_ptr()` reuse.

## 0.13.0 — 2026-11-26

- **Promote `axiom` (`DEFAULT_RECIPE = "axiom"`, matching `axiom-v5`) as the energy-optimal, clean-slate Four-Laws research default.**
  Replaces the 0.12.0 default (`unified_v12`, retained explicitly via `get_recipe("unified_v12")`) with the self-contained **`2,396`-LOC** (`<=2,500`-LOC cap) clean-slate Four-Laws core (`particlegan/_axiom/`, `particlegan/axiom.py`), cutting core complexity by **5.2×** vs `particlegan/_unified_v15/` (`12,538` LOC) while maintaining strictly zero `torch.autograd.grad`, zero SVD/`eigh`/`pinv`, strictly clock-free dynamics, exact `179` `Recipe` fields, and unchanged saved-recipe restoration (`Recipe.from_dict`):
  - **Axiom-v5 surgical fixes & host-sync reduction (`particlegan/_axiom/`, `configs/forge/ideas/axiom-v5.json`):**
    - `AxiomOptimizer`: restores Tikhonov smoothing and slow log-scale decay, replaces unconditional Critic NS with a zero-state `0`-NS Frobenius-MP fast path + compound-pathology detector + Schatten-6 spectral ceiling, switches global `p95` backtracking to batch-RMS verified trust with flip damping, and reduces GPU-CPU host syncs (`35.9 -> <=8` syncs/step).
    - `AxiomPriorRegularizer` & `AxiomCriticPenalty`: removes the `v_mu_star` deadband and variance shrink, adds measure-concentration shell separation and exact Cholesky Log-Det pullback metric (`8 < d_z <= dim`), enforces intrinsic-dimension scale-equivariant penalty scaling, and dispatches low-D / GroupNorm / quantized critics to exact RoR while keeping large Conv2d on fast `fwAD`.
    - `AxiomGANLoss` & `AxiomTransportLoss`: restores orphan macro thresholds, Weyl tie-breaking in deconflicted routing, local conformal scale, and scale-equivariant projection-complement weighting; retains input tensor storage in `_cached_manifold_gates` and `_last_raw_var_cache` to prevent cross-step `data_ptr()` / `id()` cache aliasing when PyTorch reuses freed batch buffers.
  - **Verified evidence (seed `0`, `deterministic_orthogonal`, strictly idle RTX A6000 NVML):**
    - **Canonical Forge leaderboard (`reports/forge/technique-inventory.md`, 97 certified receipts from snapshot `876222b5dde8...`):** **`45 PASS / 2 FAIL / 12 not measured`** on the 59-task fixed suite (**`102/115`** counted view requirements: `27/27` Tier 1, `75/78` Tier 2; **Screening Tier 2** on the Rev-9 comparison projection), **`53/70`** stretch assignments PASS, and **`66/78` (`84.6%`)** PASS across all measured receipts.
    - **Hardware NVML efficiency ([`rtx-a6000-v14-v15-axiom-v5-energy-benchmark.json`](reports/forge/studies/rtx-a6000-v14-v15-axiom-v5-energy-benchmark.json)):** `dcgan_performance_r256` **`32.84 ms/step` / `8.25 J/step`** solo idle (`34.39 ms/step` Forge `6/6` `dcgan_performance`, `0.0 B/param` optimizer state) — **`16%` faster / `-27%` J/step** vs `unified-v12` (`39.22 ms / 11.37 J/step`) and **`26%` faster / `-34%` J/step** vs `unified-v15` (`44.35 ms / 12.47 J/step`); `ring16_acquisition` **`556.4 J`** (`-30%` vs `unified-v15`), `grid100` **`3,511.6 J`** to first PASS (`-22%` vs `unified-v12`, `-15%` vs `unified-v15`) and **`7,023.2 J`** sustained (`-22%` vs `unified-v12`, `-27%` vs `unified-v15`).
    - **Transparent scope & trade-off disclosures:** Forge screening calibration remains provisional (`BLOCKED` under `criteria-v1.json`); `axiom` is an owner-directed research default prioritizing energy optimality and low complexity. Across all 78 measured receipts, `axiom-v5` passes `15/16` Occam (`zero_tanh_rail_escape` fails) and `66/78` overall (vs opt-in `get_recipe("unified_v15")` at `74/79` = `93.7%` and `16/16` Occam); remaining `axiom-v5` failures on drift/adaptation (`ring_hold`, `ring_extension`, `target_shift_recovery`), DDGAN chains (`model_ddgan_bimodal_chain`, `model_ae_ddgan_gaussian_chain`), `grid100_res16_tanh`, `bytegan_bigan_dictionary`, and deferred Law 1A–1B anisotropic/atomic tasks remain visible and tracked for `axiom-v6` and a dedicated short/long-horizon adaptation study.
- **Explicit `unified_v15` preset (`particlegan/_unified_v15/`, `get_recipe("unified_v15")`).**
  Adds the isolated `unified_v15` Four-Laws implementation (`44/59` fixed suite, `103/115` counted view requirements, `16/16` Occam, `74/79` measured receipts, `44.35 ms/step` / `12.47 J/step` idle RTX A6000 `r256`, `4,135.0 J` `grid100`), leaving `unified_v11..v14` bit-for-bit untouched.
- **Public training-precision & recipe-gating helpers (`particlegan`, `particlegan.training`) and Unified-v11/v12 BF16 `fwAD` hardening:**
  - Exported `float_output`, `uses_channels_last`, `uses_bf16_autocast`, `configure_bf16_autocast`, `critic_source_context`, and `uses_simplicial_manifold_loss` (with `_float_output` / `_convolution_autocast` retained as compatibility aliases) so external hosts like HyperGAN share exact `GANTrainer` BF16+NHWC and simplicial-manifold-loss parity without duplicating private capture-attribute lists, `_t2_pending_source_key` tracking, or recipe-version logic.
  - Generalized `float_output` to preserve all non-reserved Python capture attributes attached to BF16 tensors (automatically supporting current and future formulations, including `_bispf_record`, `_t1c_bypassed`, `_nash_v3_*`, and `_minimal_*`), `dict` subclass containers, and `fwAD` dual tangents.
  - Added canonical `_particlegan_bf16_autocast` flag support across probe evaluators (with legacy formulation flag compatibility) so `configure_bf16_autocast` works for arbitrary/future recipes including `unified_v14`.
  - Extended `UnifiedV11CriticPenalty._eval_image_fwad_sq` to attach the BF16 dual hook to `_InstanceNorm` layers and cleanly fall back to FP32 `fwAD` whenever BF16 is active on a non-BN critic.
  - Vectorized `UnifiedV12Optimizer`'s `stiff_gate > 0` moment update path (`_apply_slow_moment_batch_foreach`), secant `applied_deltas` rescaling, and zero-pad-copy `_sketch_grad_67_finite` flip-controller using `torch._foreach_*` (eliminating ~6,500 per-parameter CUDA kernel launches/step; 4.1x–9.8x faster stiff `opt_g.step()` on RTX A6000 with exact parity and zero `new_tensor` allocations).
- **Forge leaderboard refresh tooling, reporting source-digest isolation, and CI acceleration (PR #679, PR #680, PR #681):**
  - Added single-command multi-surface leaderboard refresh and verification (`ops/refresh_leaderboard.py`, `docs/leaderboard-refresh.md`) covering GitHub reports, Hugging Face Dataset/Space Pareto frontier, and GitHub Pages.
  - Isolated pure post-run reporting/docs/publishing modules (`NON_SCIENTIFIC_SOURCE_PATHS` in `experiments/forge/sources.py`, verified by AST import-isolation tests) so documentation/publishing edits no longer invalidate scientific `source_digest` hashes, and added cryptographically verified `--source-snapshot` promotion in `reports/forge/regenerate_technique_inventory.py`.
  - Optimized slow tests and added parallel shard workers to `scripts/ci_pytest.py`.

## 0.12.0 — 2026-11-23

- **Promote `unified_v12` (`DEFAULT_RECIPE = "unified_v12"`) as the energy-optimal 4-Law research default (PR #615, PR #620).**
  Preserves Laws 1–3 (`critic_formulation="unified_v11"`, `prior_regularizer="unified_v11"`,
  strictly `0` `torch.autograd.grad`, `0` `svd`/`eigh`/`pinv`, exact `179` `Recipe` fields,
  and explicit `get_recipe("unified_v11")` byte-identically) while introducing:
  - **Energy-optimal Law-4 optimizer (`particlegan/_unified_v12/`, `optimizer_family="unified_v12_equil"`, PR #620):**
    Replaces redundant 4D Conv2d Newton–Schulz with zero-state (`0.0000 B/param`)
    polar-matched Frobenius-MP (`torch._foreach_norm` + in-place updates), applies Law 4.4
    asymmetric `0`-NS 2D critic equilibration with automatic generator-free 2D NS fallback
    (`0` critic secant probes), adds CountSketch spectral preconditioning on stiff deep 2D
    generators, and opts into `_SUPPORTS_ZEROSYNC_V4_TENSORS = True` with strict schema-bound
    checkpoint continuation (`tests/test_unified_v12.py`).
  - **Zero-sync pullback fast-path (`particlegan/_unified_v11/`, PR #615):**
    Eliminates redundant dual-branch `torch.where` work in `unified_v11_pullback_step` and
    `update_quantizer` via single-batched control-flag syncs (`3.0` syncs/step on `r256` and
    `five_word_joint_smoke`; `12.0` syncs/step on `grid100` and `grid100_mlp16`) while
    preserving float64-reciprocal CUDA parity.
  - **Verified evidence (seed `0`, `deterministic_orthogonal`, strictly idle RTX A6000 NVML):**
    - **All-time record breadth:** **`18/18` (`100%`)** on untouched
      `r3-tier3-long-horizon-holdout-v1` (vs post-PR-#612 `unified-v11` `16/18`: passes
      `grid100_14k`, `rotated100_14k`, `staggered100`, `staggered100_14k`, `ring_extension`,
      `five_word_joint_hold`, `target_shift_recovery`, `zero_tanh_rail_escape`),
      **`46/48`** Tiers 1–2 `discriminator_stability` PASS (`17/17` T1 + `29/31` T2,
      **`29/29` required = `100%`**, `readout-5002641a64ddaa91f1fc3b87.json`), **`2/2`
      combined AE-DDGAN PASS** (`model_ae_ddgan_gaussian_chain` `paired_nmse=0.0480`,
      `t1_q0_cdf=0.2227`; `model_ae_ddgan_bimodal_chain` `paired_nmse=0.0313`,
      `t1_q0_cdf=0.1278`), **`16/16`** `r4-occam-ablation-v1`, **`6/6`** `dcgan_performance`
      (`r256 = 39.76 ms/step` median, `1,243.9 MiB`).
    - **Hardware NVML `Joules-to-PASS` (`util_before=[0,0]`, [`law4-phase2-energy-benchmark.json`](reports/forge/studies/law4-phase2-energy-benchmark.json)):**
      `dcgan_performance_r256` **`39.22 ms/step` / `11.37 J/step`** (`1.87×` faster,
      **`-45.8%`** J/st vs matched post-PR-#612+#615 `unified-v11` `73.21 ms / 20.96 J`;
      `2.19×` faster / **`-52.0%`** J/st vs pre-Phase-3 fixed `unified-v11` `85.81 ms / 23.70 J`),
      `ring16_acquisition` **`462.5 J`** (`-26.3%`), `grid100` **`4,512.9 J`** (`-36.3%`),
      `five_word_joint_smoke` **`1,398.9 J`** (`17.67 ms/step`, **`-78.8%` / `4.72×` less J**),
      and `staggered100` **`10,743.4 J`** (live-suffix PASS vs `unified-v11` FAIL).
    - **Transparent scope & trade-off disclosures:** Forge screening calibration remains
      provisional (`BLOCKED` under `criteria-v1.json`); `unified_v12` is an owner-directed
      research default selection. Standalone `t4-strange-architectures-v1` passes **`7/7`**
      (`readout-731e752ff3c4f97e29a84145.json`); combined with PR #615's float64
      `shock_t64`/`tv_now64` restoration, `grid100_res16_tanh` passes the `100k` independent
      holdout (`prec=0.9831, ctr=0.1224, score=0.6562, sat=0.0`) and `4/5` terminal checks
      with a 1-ULP step-`6250` `modes=99` dip (`6/7` combined). On `grid100_mlp16`, CountSketch
      adds `+3.1%` per-step time (`75.96` vs `73.69 ms/step`, both PASS).
- **CI & Forge evidence hygiene fixes.**
  - Preserve strict null-energy checks on historical receipts without `cost.energy` while
    validating non-null NVML `joules_total` / `energy_backend` on energy-instrumented Forge
    receipts (`tests/test_forge_hf_publish.py`).
  - Read archived `precision-stretch` preview GIFs via `git show <source_commit>:<path>` when
    evicted from the working tree (`tests/test_forge_precision.py`), and refresh compiled
    Forge reports (`ops.refresh_forge_readouts --reports-only`).

## 0.11.2 — 2026-11-21

- **Fix high-D 2D vector critic silencing, premature thermal freeze, and probe lockout in `unified_v11` (PR #591).**
  Fixes three interacting `DEFAULT_RECIPE` (`unified_v11`) failure modes on high-dimensional
  2D vector and sparse masked tabular domains (`[B, D]` with `D > 1024`, e.g. `[x * m, m]`)
  while preserving `0` `torch.autograd.grad`, `0` `svd`/`eigh`/`pinv`, zero step clocks,
  and unchanged checkpoint schemas:
  - **Confident critic non-silencing & high-D 2D vector bypass (`_t1_coupling/primitive.py`, `_unified_v11/critic.py`, `_unified_v11/loss.py`, `training.py`):**
    Adds an OT-scale-matched active-pair relativistic critic term in `T1CouplingLoss` when
    `relativistic_weight == 0.0` and `tau > 0.0` so `_sinkhorn_slack` cannot extinguish
    `∂L_G/∂d_fake` as critic margin grows (preserving exact zero Feydy bias at `tau == 0.0`),
    extends the `dim > 1024` high-D bypass from `ndim >= 4` to `ndim >= 2` with stale-capture
    `_t1c_bypassed` guarding, and keeps `ema_data_scale` / `ema_push_res` live on bypassed
    2D vectors.
  - **Prevent premature thermal freeze & cooling interactions (`_t1_coupling/critic.py`, `_unified_v11/critic.py`, `_unified_v11/prior.py`, `_unified/penalty.py`, `_unified_v11/penalty.py`):**
    Adds spread-gated under-dispersion residual in `_T1CouplingRecordMixin.ema_residual` and
    a critic-margin guard (`ema_margin`) in `_UnifiedV11RecordMixin.ema_residual` so collapsed
    masked high-D inputs stay hot while genuine equilibrium cools to `1e-6`, lifts
    `sbnp_thermal_full_max_dim` on `UnifiedV11PriorRegularizer` so `z_dim > 4` scales
    thermally when cooled, and separates `gamma_gauge` zero-mean logit anchoring from the
    Lipschitz penalty with `sqrt(s_therm)` scaling.
  - **Symmetric high-D Law 4 secant probing & no `ema_func_gain` lockout (`_unified_v11/critic.py`, `_unified_v11/optimizer.py`):**
    Removes the `dim <= 32` cap on `_d_probe_list` using the same element-budgeted slice
    (`max(2, min(16, 8192 // in_dim))`) and BN/Dropout-safe eval baseline as generator
    probing, and makes `free_ratio` / `step_active_probe` scale-aware in
    `pre_gain_applied * step_scale * th_tikh * coh_eff` with `1.15 * ema_func_gain`
    hysteresis and a `1e-4` equilibrium floor.
- **Public Unified-v11 composite generator probe capture & RNG-safe replay (`particlegan.unified_v11`).**
  - Export `record_unified_v11_generator_probe` (with recursive nested container/dict slicing for conditioning inputs and eval-mode baseline capture for BatchNorm/Dropout subgraphs) and `UnifiedV11EquilibratedOptimizer.set_generator_probe_parameters(...)` so composite multi-component generators (such as HyperGAN `ComponentGraph` models) can register full-graph secant probes without mutating checkpoint schemas.
  - Preserve CPU/CUDA/Python/NumPy RNG states in addition to per-submodule `.training` modes and BatchNorm running statistics during Unified-v11 probe replay (`_eval_v11_probe_forward`).
  - Expose transient runtime probe-consumption counters (`last_probe_consumed`, `probe_consumptions`) on `UnifiedV11EquilibratedOptimizer` without changing optimizer `state_dict()` schemas.
- **CI integration optimization (PR #590).**
  Reduce duplicate CI runs and balance regression shards.

## 0.11.1 — 2026-11-20

- **Fix autograd-hook VRAM leak in live-endpoint critic captures (PR #585).**
  Break the closure reference cycle across all six critic hook sites (`_unified_v11`,
  `_unified_v4`, `_unified`, `_t2_critic`, `_nash_v3`, `_minimal`) where `_live_captures`
  entries held strong references to autograd-attached `unscaled` logit tensors via
  `register_hook` lambdas, causing linear GPU memory growth during long training runs
  (e.g., `920 -> 3,898 MiB` over 800 CIFAR-10 steps). Adds CPU & CUDA weakref / flat-VRAM
  regression tests (`tests/test_unified_critic_vram_leak.py`), flattening peak VRAM to
  `953–968 MiB` across 2,000 steps.
- **Universal 1st-order `fwAD` Laws 2–4 and function-space equilibrated optimizer for `unified_v11` (PR #587).**
  Fixes `unified_v11`'s 4D high-resolution image failure while preserving `0`
  `torch.autograd.grad`, `0` `svd`/`eigh`/`pinv`, exact `179` `Recipe` fields, and untouched
  frozen `_unified/` & `_unified_v4/` baselines:
  - **Law 2 (`particlegan/_unified_v11/penalty.py`):** Replaces the `dim <= 1024` and
    `dim > 1024` finite-difference image secant branches (BF16 ULP underflow at
    `img_delta=0.02`, ReLU `O(1/delta)` kink impulses, mixed real+fake BN stat shift, and
    6× over-strong logit gauge) with one universal 1st-order `fwAD.dual_level()` per-sample
    Born-Infeld critic penalty (`live_endpoints_reused=True`, `eff_floor=0.0005/dim`,
    `eff_gauge=min(gamma_gauge, 0.005)`, temporary BN `eval()` + BF16 dual-tangent cast
    hook `_bn_dual_bf16_hook`).
  - **Law 3 (`particlegan/_unified_v11/loss.py`, `critic.py`):** Keeps `ema_margin`,
    `ema_data_scale`, and `ema_push_res` live across all tensor ranks without caching 4D
    images in `_t1c_capture`, and retains `fibre_rel_blend=0.50` scalar logit-space
    relativistic coupling (`softplus(f - r)` / `softplus(r - f)`) when `loss == "non_saturating"`
    and pixel-space `t1c` capture is skipped (respecting explicit user `loss` overrides).
  - **Law 4 (`particlegan/_unified_v11/critic.py`) & Universal Equilibrated Optimizer (`optimizer.py`):**
    Removes syntactic `L_count` / `deep_gate` and `ndim != 2` / `dim > 32` generator probe
    gates; replaces them with an element-budgeted, BN-safe generator secant probe driving
    checkpointed envelope-tracked `_ema_func_gain` + generator-only `stiff_gate`, degree-0
    `peak_rms` batched Tikhonov smoothing, exact `1×1` Conv2d `sqrt(out_ch / in_ch)` scaling,
    and lifted `min(r, c) <= 512` per-singular-value Tikhonov polar.
  - **Checkpoint compatibility note:** `UnifiedV11EquilibratedOptimizer` now checkpoints
    `_ema_func_gain` and per-parameter `peak_rms` with strict validation and explicitly
    rejects older `0.11.0` optimizer checkpoints rather than silently reconstructing missing
    history.
  - **Verified evidence (seed `0`, zero recipe overrides):** On real CIFAR-10
    (`benchmarks.cifar_toy`, `B=128`), `unified_v11` drops FID from `2730.72 -> 458.95`
    (`N=200`, `s800`, `TV=0.109`, best `436.99` @ `s2000`, `8.22–8.49 it/s`, `953 MiB`) and
    `2618.93 -> 728.10` @ `s800` -> `432.43` @ `s2000` (`N=16384`, `10/10` cov,
    `8.17–8.56 it/s`, `967 MiB`, `1.67×–2.18×` faster than `empf`). Official 7,000-step
    100-Gaussians gate passes at `100/100` modes, `hq = 99.025%` (`+0.50%` over `0.11.0`),
    `mode_tv = 0.0436`. Forge `discriminator_stability` Tiers 1–2 reaches `25/27` required
    PASS (`11/11` T1, `14/16` T2 — `+1` net required gain over `0.11.0`'s `24/27`, gaining
    `five_word_joint_hold` and `staggered100`, with `39/46` total graded PASS and 7
    non-passes transparently disclosed: `model_ddgan_gaussian_chain`,
    `model_ae_ddgan_gaussian_chain`, `vector_unequal_width`, `interp_sparse_arc`,
    `interp_img_patch_shift`, `interp_saddle_barycentric`, `t3_iso_scale_mass_8x`).
    Historical `0.11.0` Forge receipts remain untouched.
- **Reproducible CIFAR-10 Transformer-G / ResNet-18-feature-D benchmark (PR #586).**
  Add `benchmarks/cifar_toy/` and `tests/test_cifar_toy_benchmark.py` (registered in
  `experiments/forge/history.py`) with SHA-256-verified frozen ImageNet ResNet-18 critic and
  independent ResNet-34 Feature-FID/KID/coverage evaluator, pure public API, and matched
  cross-trainer provenance digests.

## 0.11.0 — 2026-11-20

- **Default Unified 4-Law research selection (`DEFAULT_RECIPE = "unified_v11"`).**
  Promote `unified_v11` (matching [`unified-v11`](configs/forge/ideas/unified-v11.json),
  [`particlegan/unified_v11.py`](particlegan/unified_v11.py)) as the public default
  trainer preset while preserving explicit `get_recipe("minimal")`, `get_recipe("empf")`
  (`empf-v8`), `get_recipe("rhbi_optimized")` (`rhbi-v12`), `get_recipe("rhbi")`
  (`rhbi-v1`), `get_recipe("bispf_optimized")`, and `Recipe.from_dict(saved_fields)`
  historical restoration (`particlegan/recipe_compat.py` untouched). `unified_v11`
  runs with strictly **`0` `torch.autograd.grad`, `0` `svd`/`svdvals`, `0` `eigh`/`eigvalsh`,
  and `0` `pinv`** across every forward, backward, penalty, transport, conditional fiber
  bank, prior, and optimizer path, uniting all four GAN laws with fused `Z=[X,Y]` Quintic
  Tikhonov polar equilibration, zero-`autograd.grad`/zero-`pinv` analytic `direction_blend`,
  zero-`eigh` cyclic-Jacobi cross-batch conditional fiber operator matching, and
  scale-invariant `k=4` density-ratio orphan macro-transport. Across `9` concluded Forge
  studies on `NVIDIA RTX A6000` (seed `0`, see
  [`reports/forge/studies/r4-solution-complete.md`](reports/forge/studies/r4-solution-complete.md)
  and the [technique inventory](reports/forge/technique-inventory.md)), `unified-v11`
  leads the canonical leaderboard at **`95/115` counted view requirements** (`27/27`
  Tier 1, `68/78` Tier 2), **`40/69` full-suite passes** (`6` FAIL, `11` not measured,
  `12` blocked), **`50/68` stretch view assignments**, and **`41/47` Tiers 1–2
  `discriminator_stability` PASS** (`55/62` deduplicated unique runnable tasks PASS;
  `90.24 ms/step` median and `1,372.60 MiB` allocated VRAM on matched 66-cell
  `dcgan_performance_r256`, `89.19 ms/step` in `readout-1e5fb235`), including `11/11`
  required Tier 1, `3/3` precision (`fp32`/`mxfp8`/`nvfp4`),
  `model_ddgan_gaussian_chain`, `3/3` extreme Nash imbalance, `2/2` causal isolation, `5/5`
  shift/rail recovery, `3/3` `7.5k` ring endurance, `6/7` 16-layer strange architectures,
  `15/17` Tier-3 `14k` holdout (`grid100_14k` and `rotated100_14k`), and `3/4` runnable
  Tier-4 tasks — ahead of `empf-v8` (`83/115`, `23/69`, `129.72 ms/step`) and `rhbi-v12`
  (`83/115`, `33/69`, `590.59 ms/step`). Forge screening calibration remains provisional
  (`BLOCKED` under
  `criteria-v1.json`; see [`reports/forge/studies/r3-calibration-holdout.md`](reports/forge/studies/r3-calibration-holdout.md)),
  and remaining failures (`staggered100` / `staggered100_14k`, `grid100_res16_tanh`,
  `five_word_joint_hold`, `model_ddgan_bimodal_chain`, `model_ae_ddgan_gaussian_chain`,
  `interp_img_patch_shift`, `interp_saddle_barycentric`, plus `12` unimplemented Tier-4
  stretch hosts) stay explicitly disclosed.
- **Minimal-Principle `<900`-LOC fast core (`preset="minimal"`, `particlegan/minimal.py`).**
  Add the self-contained `899`-LOC, zero-sibling-import core (`minimal-v2`) with zero
  `autograd.grad`, zero SVD/`eigh`/`pinv`, and zero Newton-Schulz (`55.50 ms/step` and
  `1344.84 MiB` peak allocated VRAM on `dcgan_performance_r256`, `6/6` PASS; see
  [`reports/forge/studies/r4-minimal-core.md`](reports/forge/studies/r4-minimal-core.md)).
- **The Four Laws of Stable, Energy-Efficient GAN Convergence ([`docs/four-laws.md`](docs/four-laws.md)) & Occam's Razor causal ablation (Rounds 1–4).**
  Document the four laws proved across Rounds 1–4 — Law 1 (Measure & Prior Topology: `t1_prior`, `t1_coupling`), Law 2 (First-Order Zero-`autograd.grad` Critic Geometry: `t2_secant`), Law 3 (Scale–Mass–Fiber Gauge Identifiability: `rgdt`, `cond_fiber`), and Law 4 (Function-Space Two-Player Equilibration & Depth/Rail Trust: `t4_nash`, `t4_nash_v3`, `t4_arch`) — together with unified synthesis (`unified_v4`..`unified_v11`), bounded Tier-3/Tier-4 calibration holdout audits (`r3-calibration-holdout`), and Round-4 Track 2's 8-arm Occam's Razor causal knockout study (`r4-occam-ablation-razor`, showing the four laws collapse into Two Mathematical Halves with depth-gated 2D Newton-Schulz strictly on `L > 5` skip-free MLPs while identifying redundant epicycles).
- **Automated release pipeline from `master`.**
  Publish new versions automatically to PyPI via Trusted Publishing on `master` merges and
  publish a verified GitHub source release at the tested commit after PyPI succeeds
  (`.github/workflows/release.yml`, `scripts/publish_github_source.py`).

## 0.10.1 — 2026-11-16

- **Default EMPF research selection (`DEFAULT_RECIPE = "empf"`).**
  Promote `empf` (matching [`empf-v8`](configs/forge/ideas/empf-v8.json)) as
  the public default trainer preset while preserving explicit
  `get_recipe("rhbi_optimized")` (`rhbi-v12`), `get_recipe("rhbi")` (`rhbi-v1`),
  `get_recipe("bispf_optimized")`, and `Recipe.from_dict(saved_fields)`
  historical restoration. In the
  [technique inventory](reports/forge/technique-inventory.md), `empf-v8` ties
  `rhbi-v12` at `83/115` counted view requirements (`27/27` Tier 1 PASS,
  `56/78` Tier 2, `0/10` Tier 3) with `23/64` fixed-suite passes (`17` FAIL,
  `8` not measured, `16` blocked), while scaling cleanly on 32×32 image-scale
  CUDA workloads (`newton_schulz_hybrid_v1`, `batched` SVD, and
  `bf16_channels_last`). Forge screening calibration remains `BLOCKED`, Tier 2/3
  full-tier qualification is incomplete, and untouched holdout confirmation
  remains open.

## 0.10.0 — 2026-11-16

- **Default RHBI research selection (`DEFAULT_RECIPE = "rhbi_optimized"`).**
  Promote `rhbi_optimized` (matching [`rhbi-v12`](configs/forge/ideas/rhbi-v12.json))
  as the public default trainer preset while preserving explicit `get_recipe("rhbi")`
  (`rhbi-v1`), `get_recipe("empf")`, `get_recipe("bispf_optimized")`, and
  `Recipe.from_dict(saved_fields)` historical restoration. In the
  [technique inventory](reports/forge/technique-inventory.md), `rhbi-v12` leads
  screening at `83/115` counted view requirements (`27/27` Tier 1 PASS, `56/78`
  Tier 2 with 2 not run, `0/10` Tier 3 with 10 not run), `33/64` fixed-suite
  passes (`7` FAIL, `8` not measured, `16` blocked), and `35/63` stretch view
  assignments. Screening calibration remains `BLOCKED`, Tier 2/3 full-tier
  qualification is incomplete, and untouched holdout confirmation remains open.
- **Public RHBI and EMPF families (`particlegan.rhbi`, `particlegan.empf`).**
  Export `RHBILoss`, `RHBIGANLoss`, `RHBICriticPenalty`, `RHBICriticAdam`,
  `RHBIStepRecord`, and `bind_rhbi_record` alongside `EMPFPriorRegularizer`,
  `EMPFCriticPenalty`, `EMPFCriticAdam`, and `EMPFStepRecord`. `RHBIGANLoss`
  pairs calibrated critic log-odds with debiased slack-screened Kantorovich-Hellinger
  Sinkhorn generator transport (`g_loss(fake_logits, real_logits)` requires
  `real_logits` when `rhbi_weight > 0`), conditional coordinate-whitened
  IMQ-CMMD fibre transport with residual step damping, sparse-mode support
  completion, and discrete-simplex / 1D-curve chart settling.
- **DualNorm dense polar profiles (`polar_cpu_v1`, `newton_schulz_v1`, `newton_schulz_hybrid_v1`).**
  Add CPU-offloaded small-matrix polar SVD (`polar_cpu_v1`) and guarded CUDA
  Newton-Schulz polar profiles (`newton_schulz_v1` and `newton_schulz_hybrid_v1`
  for dense matrices with `min(shape) >= 64` and `max(shape) >= 128`), plus
  discrete-simplex Chamfer hold cooling in `EMPFCriticPenalty`.
- **Zero-sigma particle interpolation stretch benchmarks and release workflow fix.**
  Register the three zero-sigma `ParticlePrior` geometric interpolation stretch
  benchmarks (`interp_sparse_arc`, `interp_saddle_barycentric`, `interp_img_patch_shift`)
  in the 64-test full-suite leaderboard and allow `HyperGAN/ParticleGAN` alongside
  `255BITS/ParticleGAN` in `.github/workflows/release.yml`.

## 0.9.8 — 2026-11-09

- **Tiny PyPI source distribution (`sdist`) and strict distribution gates.**
  Prune non-package trees (`tests`, `reports`, `results`, `experiments`,
  `benchmarks`, `lib`, `configs`, `data`, `runs`, `artifacts`, `docs`,
  `examples`, `ops`, `scripts`) and workspace files via `MANIFEST.in`,
  shrinking `particlegan-0.9.8.tar.gz` from `1.7 MB` (`7.14 MB` uncompressed)
  to `~343 KiB` (`~373 KiB` wheel) and enforcing `< 600 KB` byte-size and root
  allowlists in CI and packaging tests.
- **Reviewed model objectives, pre-subtraction `NaN` masking, and public exports.**
  Export `conditional_energy_score`, `joint_reconstruction_loss`,
  `masked_mixture_nll`, `NeuralComponentImputer`, `masked_quantile_transport_loss`,
  `EmulatedLinear`, and `emulate_linear_training_` from top-level `particlegan`.
  Mask `observed` via `torch.where` before subtracting `components` in
  `masked_mixture_nll` so unobserved `NaN` entries cannot contaminate gradients
  when `observed.requires_grad` is `True`.
- **Formula-faithful recipe admission, independent critic histories, and provenance.**
  Enforce formula-faithful admission across vector, BiGAN, DDGAN, VAE-GAN, and
  MisGAN hosts (including rejecting unsupported `model="ddgan"` and
  `training_precision="bf16_channels_last"` settings on MisGAN hosts before
  optimizer construction and isolating independent MisGAN critic histories),
  separate faithful numerical `FAIL` from `BLOCKED`/`INCOMPLETE`/`UNMEASURED`
  outcomes, and preserve byte-exact historical receipts with ancestral
  source-binding verification
  ([integration scope](reports/forge/reviewed-pr-integration-20261109/README.md),
  [audit](reports/gpt-audit-2026-11-08.md)).
- **Exact trainer checkpoint continuation with gradient response.** Save,
  validate and restore gradient-coherence history when enabled, preserving the
  next update in fresh and previously used trainers. Reject older enabled-response
  checkpoints without that history instead of silently restarting the controller.
- **Completed-update checkpoint boundaries.** `GANTrainer.state_dict()` and
  `load_state_dict()` reject active updates before changing live state, including
  calls inside the public `generator_real` callback.

## 0.9.7 — 2026-11-05

- **Unified `stretch` experiment role and tier-by-tier stretch leaderboard.**
  Introduce `stretch` (`STRETCH_ROLES = {"stretch", "held"}`) across
  `experiments.forge` and `particlegan_forge` as the canonical role unifying
  extra-credit, optional, frontier, and held benchmark experiments. Render a
  dedicated `## Stretch experiments` table by tier (`Tier 1 stretch` through
  `Tier 4 stretch` plus `Total stretch`) in the
  [technique inventory](reports/forge/technique-inventory.md) and clarify
  view-level `ranking` versus task-assignment roles (`required`, `stretch`,
  `diagnostic`) in [RESEARCH_GLOSSARY.md](RESEARCH_GLOSSARY.md) and
  [EXPERIMENTATION.md](EXPERIMENTATION.md).

## 0.9.6 — 2026-11-05

- **Default BISPF winner (`bispf_optimized`) and Tier 1/Tier 2 qualification.**
  `DEFAULT_RECIPE = "bispf_optimized"` (`bispf-optimized-quintic-v1`) is the
  selected default trainer preset. All seven runnable selected families pass
  all `27/27` required Tier 1 smoke tasks on the current source cohort, and
  both BISPF and BCAP reach `77/78` required Tier 2 tasks (`105/115` and
  `109/115` total across Tiers 1–3) in the refreshed
  [technique inventory](reports/forge/technique-inventory.md).

## 0.9.5 — 2026-11-05

- **Explicit BISPF input domains.** Declare `input_domain="continuous"` for
  native feature vectors of any width, or `"categorical"` for categorical
  matrices, through the public critic penalty factory or `GANTrainer`'s
  `penalty_options`. The default `"auto"` preserves the existing shape gates;
  an explicit domain overrides automatic simplex detection. Small categorical
  codebooks also stay out of the late continuous 2D thermal-freeze rule.
  Checkpoints retain the selected domain and reject mismatched restoration.
- **Runnable categorical examples.** The
  [input-domain guide](docs/bispf-input-domains.md) includes a checked arithmetic
  walkthrough, four-category GAN training with numerical quality gates and an
  actual-training GIF, and exact checkpoint continuation. The checked CPU run
  passes its final five observations; it supplies no Forge qualification or
  native-model quality claim.

- **One recipe selection path for new training.** Bare or partial `Recipe(...)`
  calls now raise a migration error rather than silently selecting historical
  KA2 settings. Use `get_recipe()` for the selected public default, or explicitly
  select a named preset. Restore compact saved configurations with
  `Recipe.from_dict(saved_fields)`; historical values and checkpoint semantics
  are preserved. Fully resolved field dictionaries and `dataclasses.replace`
  remain supported. This is an intentional constructor API break, not a change
  to any preset, optimizer, penalty or recorded training result.

## 0.9.4 — 2026-10-23

- **Python 3.11+ requirement and patched build/image dependencies.** Raise
  `requires-python` to `>=3.11`, require `setuptools>=84.2.1` in
  `[build-system]` (`GHSA-8rrh-rw8j-w5fx`), and require `Pillow>=13.0.0,<14`
  across the `lunar`, `images` and `experiments` optional dependency groups
  (`CVE-2026-112900`, `CVE-2026-112393`, `CVE-2026-112403`), removing the
  Python `< 3.11` `tomli` fallback. Core `particlegan` runtime dependencies
  (`torch>=2.6,<3`, `numpy>=1.26,<3`) are unchanged.
- **`UpdatePolicy` extra-state validation.** `UpdatePolicy._check_tensors`
  validates module `_extra_state` metadata (such as `MoGParticlePrior`
  `sigma_rel` and `standardize` fields persisted via `get_extra_state`) before
  restoring model or optimizer state.
- **Expanded Forge model toys and preserved diagnostic records.** Register
  public AE, VAE and DDGAN research model toys in the Forge view-v14 inventory
  and preserve source-bound saved-state VAE/DDGAN failure diagnostics and
  concluded R1/R2 and stall-reach/G-idle recall archives.

## 0.9.3 — 2026-10-18

- **Opt-in guarded FP64 Gram and diagnostic SVD dense profiles.** Add
  `optimizer_dense_profile="gram_fp64_guarded_v1"` and
  `optimizer_dense_profile="gram_fp64_svd_diagnostic_v2"` for FP32 dense
  DualNorm matrices: FP64 Gram/`eigh` with the original FP32 cutoff, absolute
  `.001` smoothing, a guard on the factors actually used, and device-local
  FP64 SVD fallback (`gram_fp64_svd_diagnostic_v2`) when the Gram rank cutoff
  is ambiguous. Wide matrices (`rows < columns`) orient along the shorter
  dimension `(rows, rows)` before forming the Gram matrix. Default, convolution,
  vector and prior laws are unchanged. Selected checkpoints require the same
  profile. See the [contract and evidence](reports/gram-profile/README.md) and
  [diagnostic SVD profile docs](docs/gram-svd-diagnostic-profile.md).

## 0.9.2 — 2026-10-18

- **SBNP prior regularizer degree-of-freedom guard in high-dimensional latents.**
  `SBNPPriorRegularizer` and `sbnp_regularization_loss` now only include the
  latent dimension `d` in `min_dof = 2 * k_nn + (d if d <= 4 else 0)` when
  operating in the `d <= 4` full-covariance regime. High-dimensional latents
  (`d > 4`, where only marginal variance floor, normalized off-diagonal
  covariance Frobenius norm, and `k_nn` repulsion are evaluated) remain active
  at standard batch sizes (`n = 16, 32, 64`) and when `num_particles > 1024`
  passes sampled unique batch rows from `GANTrainer._step`.

## 0.9.1 — 2026-10-17

- **Faster BISPF execution is the public default.** `bispf_optimized`,
  `get_recipe()` and model-family factories now select batched convolution
  updates and BF16/channels-last training for supported CUDA convolutional
  networks. Parameters, optimizer arithmetic, losses and sampling stay FP32.
  CPU, nonconvolutional and small networks retain their previous precision;
  CUDA devices without native BF16 support also use FP32 network execution.
- **Matched throughput:** 2.15× at 256×256 DCGAN, 2.14× at batch 64 and about
  2.3× on the small Forge image tasks versus the 0.9.0 quintic default.
  The six DCGAN workloads improve 1.02–2.15×, with a 1.66× geometric mean.
  All 19 incumbent numerical passes and 11 existing failures are preserved.
  Separate endurance controls pass; current revision-12 Tier 3 remains
  blocked by existing required failures. This owner-directed default choice
  adds no calibration or natural-image quality claim. See the
  [source-bound readout](reports/forge/bispf-throughput/README.md).
- **Restoration:** persist and restore complete recipe fields. To select the
  0.9.0 execution policy explicitly, use
  `get_recipe("bispf_optimized", optimizer_svd_backend="native", training_precision="float32")`.
  Native and batched optimizer checkpoints reject mismatched backends; CUDA
  graphs are disposable and rebuilt after restoration. Other named trainer
  presets and bare `Recipe()` retain their earlier settings.

## 0.9.0 — 2026-10-17

- **Optimized BISPF is the public recipe default.** `get_recipe()`, `"gan"`
  and model-family factories now select `"bispf_optimized"`, matching the
  selected Forge recipe: non-saturating loss, Born-Infeld critic penalty,
  SBNP prior regularization and zero-momentum DualNorm at constant rates.
  Large annotated CUDA convolution kernels use the quintic update; CPU,
  tiny kernels, dense matrices and sampled prior rows keep their native laws.
  The owner-directed promotion adds no calibration or robustness credit.
  Existing `"bcap"`, `"bispf"`, `"ka2"` and other explicit presets remain
  available; `Recipe(**saved_fields)` retains its restoration semantics.
- **Dependency-light recipe fields.** `get_recipe_fields` exposes resolved
  preset settings for applications such as HyperGAN without constructing
  networks or optimizer state.
- **Module ownership for custom optimizer groups.**
  `recipe.make_generator_optimizer(groups, modules=...)` binds convolution
  layouts from ordinary PyTorch modules while preserving caller learning rates
  and prior roles. Frozen parameters are excluded from this module-bound path;
  no architecture-specific training wrapper is required.
- **Complete BISPF observer checkpoints.** Normalized and plain-Adam BISPF
  optimizers now save transport residual/data-scale moving averages and the
  exact thermal scale, fixing divergence after restoring a training checkpoint.
  Earlier packets that omitted these moments fail with a compatibility error
  directing callers to their original package source. New formulation-optimizer
  packets also retain the exact thermal scale; historical formulation packets
  keep their earlier reconstruction behavior. BCAP and KA2 observer formats
  remain unchanged.
- **Measured optimization scope:** the frozen optimization cohort preserves
  all 19 passing task verdicts and its 11 existing failures. The current
  overview's 81/81 counts repeated view requirements from 18 distinct tests.
  Wide synthetic image diagnostics exercise the accelerated path; these do
  not establish CIFAR quality or transformer acceleration. See the
  [complete readout](reports/forge/bispf-optimized/README.md).
- **Release checks use the validated tagged commit.** Develop/master pull
  requests and pushes run CPU CI; published releases run the complete existing
  matrix and package checks regardless of the release target branch label.
  The publisher uploads only the artifacts produced by those checks.

- **Learned-noise floor release:** exclude the noise group's own stationarity
  scale from the model/table settlement gate, preventing clamped noise from
  holding its floor high indefinitely. E22 and `e22_routed` share this repair;
  noise learning-rate control and checkpoint/configuration formats are retained.
  CPU/CUDA regressions cover a real frozen verdict, resumed learning, exact
  recovery and served noise after release.
- **Many-site synchronization:** DV12 retains only the last two detached
  applications and computes their diagnostics on demand. Accelerator routing
  batches finite checks once per complete forward; CPU validation stays eager.
  Checkpoint format, training noise, row controls and R1 defaults are unchanged.
  The 71-site CUDA fixture reduces forward-pair scalar reads from 1,014 to 22,
  with exact before/after recovery state. See the
  [measurement and integration notes](docs/e22_routed_readbacks.md).
- **Routed game units and mass invariance:** the pooled-token example applies
  KA2 through a contextual token penalty view; RpGAN scores stay pooled.
  Routed DV12 ignores inactive rows and aggregates duplicate-position mass
  for bandwidth and support. Selected serving copies use their matching
  routed row buffers. Checkpoints record `mass_atoms_v1`; older raw-row-law
  routed checkpoints require the prior release or a new training trajectory.
  Example checkpoints record penalty units and explicit feature-harm bounds.
  New private-RNG candidate probes report noisy game gradients, and matched
  longer comparisons retain frozen/movable controls. These diagnostics do not
  select moves or certify game repair. See the
  [qualification guide](docs/e22_routed_game.md).
- **R1 is the E22 default** ([docs](docs/r1-rotation.md)): the E22 preset (and
  `e22_routed`, `configs/100gaussians/e22-noout.json`) sets
  `reopen_signal="optimizer"` (an Adam-memory surprise detector that, on an
  abrupt rise, re-opens the LR ladders and rescales the shocked second moments)
  and `reopen_anchor="release"` (the re-open lets KA2's EMA-critic anchor
  release). It passes the native rotation test 3/3 (E22 before R1: 1/3) and
  the routed moving-target example; on static targets it does not fire and
  matches E22 exactly. `reopen_signal="none", reopen_anchor="hold"` restores
  the previous behaviour. `examples/e22_routed_moving.py` runs the routed
  paired-edit example with a turning target.
  `e22_routed_sites.make_loop` and `e22_routed_support.make_loop` accept
  `recipe_overrides=`; `reports/r1-integrations` checks R1 in every E22
  integration loop.
- **Composable E22 policy:** `get_recipe("e22", **task_overrides)` ships the
  preset in the installed package. `E22Policy` / `UpdatePolicy` expose the
  same lifecycle controls used by `GANTrainer`, explicit optimizer/table
  ownership and callbacks, recovery state and served-model snapshots. The
  independent formulation retains its unconditional particle contract. See the
  [external-loop and migration guide](docs/e22.md).
- **Conditional routed paired adaptation:** `get_recipe("e22_routed", ...)`
  supports dense softmax banks through explicit `RoutedRows` and `RoutedBatch`
  inputs. Row controls use conditional counterfactual evidence and guarded
  whole-bank moves, including routing state, averages and optimizer/controller
  recovery. Clean served forwards preserve the selected conditional bank.
  Frozen BF16 modules can accompany FP32 trainable tensors; averages copy
  frozen parameters exactly. See the [design and validation](docs/e22_routed.md).
- **Shared-bank token routing:** a full-model `RoutedRows` callback supports
  multiple sequential routing sites using one particle bank and controller.
  Candidates rerun downstream routing and guard the final paired output.
  Usage averages over tokens within each context and then over sites.
  Routed evidence now refreshes when birth/death is disabled; disabling both
  row controls also permits a frozen bank. See the
  [site contract and spatial comparison](docs/e22_routed_sites.md).
- **Routed split recovery and probe cost:** split rows inherit half the parent
  first Adam moment and one quarter of its second/AMSGrad moments, preserving
  optimizer age and avoiding a late-split update spike. `RoutedRows` accepts
  checkpointed `probe_interval` (default 1), independent of context eligibility,
  and an optional `output_error_guard` for fast and averaged paired-output MSE.
  Per-update gradient observations continue between probe passes.
- **CUDA learned-noise initialization:** repair the rounding case where
  `exp(log_sigma)` falls below the physical floor while `log_sigma` is at or
  above its log floor. The initial gradient stays active; ordinary floor
  gradients, native FP64 behavior and checkpoint parameter layout are retained.
- **Explicit KA2 critic formulation** ([docs](docs/ka2.md)) replaces
  K3P for the retained KA2 preset. Loss, optimizers, schedules and noise are unchanged. The critic penalty
  is RMS R1 plus a fake cap for its first 799 calls, then an even blend with
  one-sided caps and an EMA-critic gradient anchor. The anchor's gate and the
  EMA critic's decay (`reg_anchor_min_decay`, default .9, replacing
  `reg_anchor_decay`) follow the critic's own Adam moment surprise instead of
  its learning rate, with guarded EMA reseeds. `recipe.name` is `"ka2"`.
  `critic_r1_real=False` drops the R1 term.
- **Trainer checkpoints use schema 4.** Schema 1–3 checkpoints come from older
  formulations and raise `ValueError`; resume them with the release that wrote
  them (0.8.0 for K3P). Checkpoints that record removed fixed choices or
  `initialization` still load.
- **Opt-in controllers, all off by default:** a continuous LR controller
  (`continuous_policy`, `lr_control="stationarity"`), learnable output noise
  (`output_noise_mode`), AMSGrad (`amsgrad`), particle birth-death
  (`particle_birth_death`, `birth_death_space`, `birth_death_feature_scale`,
  `birth_death_isolation`), per-row gradient evidence (`row_evidence_*`), the
  table release rule, a served averaged model (`serve_average`) and
  `reopen_signal`.
- **[E22](docs/e22.md)** (`configs/100gaussians/e22-noout.json`) switches them on:
  a schedule-free configuration for the native 100-Gaussian problems with no
  data-space statistic in its control path. It passes grid100, rotated100 and
  staggered100 (precision .982–.985), the ×0.75/×1.33 LR stress, 14k and 28k
  updates, and the 13-gate portability suite.
- K3P replays through `benchmarks.legacy.recipe` ([docs](docs/k3p.md#replaying-k3p)).

- **`GANTrainer.sample()` returns clean samples by default.** It previously
  added the current training output noise; pass `output_noise=True` for that.
  Training is unchanged (sampling uses only its own stream). The toy100
  benchmark now scores clean evaluation and holdout draws too (its summary
  records `"eval_output_noise": "clean"`), so its scores are not directly
  comparable with earlier runs trained with output noise.
- **Explicit initialization API, `particlegan.init`**, in the style of
  `torch.nn.init`. `init.deterministic_orthogonal_(module, *, seed=0,
  strict=True)` gives trainable weights deterministic orthogonal matrices at
  PyTorch's default scale and patterned biases, particle tables an R2 cloud
  (MoG spacing recalibrated), and a `BatchDistanceDiscriminator` zero initial
  batch-feature coefficients. It hashes `seed`, so it consumes no RNG; the
  examples use G=0, D=1, E=2. Layers declare their parameters with
  `init.register(cls, {name: Uniform/Normal/R2Normal/KEEP})`, and undeclared
  trainable parameters raise unless `strict=False`; `init.declarations(module)`
  shows what a network would get. NumPy is now a core dependency. See the
  [API reference](docs/api.md#initialization).
- The recipe, `make_prior`, `make_optimizers` and `GANTrainer` never change
  weights; `Recipe.initialization` and `initialize_` from earlier development
  builds are removed (`initialize_(m, key=k)` is
  `init.deterministic_orthogonal_(m, seed=k)`). `GANTrainer` still loads
  checkpoints whose saved recipe records `initialization`.
- The research initializer registry, including the exact historical
  `batch_feature_zero` replay hook, moved to `benchmarks/init_research`
  (`python -m benchmarks.init_research.init_registry --init NAME -- script.py`).
  Its report documents the 22/22 fixed-suite result, passing long hold,
  unresolved shifted-target recovery, and mathematical limits.

## 0.8.0 — 2026-09-25

- **K3P is the default and only formulation.** The critic penalty blends R1 +
  fake-gradient cap into one-sided gradient caps plus an EMA-critic gradient
  anchor as the critic LR decays; the recipe optimizers add the critic spike
  guard, sparse latent-row damping and direct-particle response. It replaces
  GAN v3, which lost quality over longer training runs.
- **Plain PyTorch loop.** Build everything through the recipe:
  `recipe.make_optimizers(G, D, prior, ema_critic=...)`,
  `recipe.make_critic_optimizer(D2, ema_critic=...)` for extra critics, and
  `penalty = recipe.make_critic_penalty(opt_d)`; then
  `d_loss = adv + penalty(D, real, fake, *cond)` and the usual
  `zero_grad()/backward()/step()`. All regularization state lives in the
  optimizers' `state_dict()`s, so standard checkpoints resume exactly.

- **One formulation, no technique menu.** Removed from `particlegan`: the
  `Recipe` fields `loss_type`, `gan_mode`, `reg_arm` and `reg_method`; the
  non-K3P penalty arms, norms, center annealing and finite differences of the
  critic gradient penalty; the hinge/Wasserstein/LSGAN/vanilla/RaGAN modes of
  `GANLoss` (now RpGAN logistic only); `recipe.make_gradient_penalty`; the
  unused `generator=` of `make_critic_penalty`; the `particlegan.locked_shared`
  stamp; and the `GradientPenalty` top-level export. Training with the default
  recipe is bit-identical. `GANTrainer` still loads checkpoints whose recipe
  recorded the removed fields at their only supported values.
- Benchmarks that replay archived GAN v3 / locked_shared / arm-study
  configurations now resolve them through frozen copies in `benchmarks/legacy/`
  (`LegacyRecipe`, the multi-arm penalty, the multi-mode loss, the stamp).
- Removed the arm-study drivers and configs (`experiments/train_arm.py`,
  `gen_configs.py`, `compare_priors.py`, `make_video.py`, `provenance.py`,
  `probe_cifar_fd.py`, top-level `configs/*.yaml`, `configs/audit/`), the
  `lib/gan_loss.py` / `lib/grad_regularizers.py` shims, the docs for GAN v3,
  locked_shared, prior controls and the develop API review, and tests that
  only asserted removed behaviour.
- Added ablation switches `Recipe.reg_anchor_weight` (1.0; 0 removes the
  critic penalty's EMA-anchor term) and `Recipe.direct_particle_gain` (True;
  False keeps the direct-particle LR unscaled).
- Rewrote the README for newcomers around a runnable plain-PyTorch loop
  (executed by `tests/test_readme.py`).

- **Every repository trainer uses the default formulation.** Examples,
  experiments and `lib/` trainers build their optimizers and critic penalties
  only through `recipe.make_optimizers` / `make_critic_optimizer` /
  `make_generator_optimizer` / `make_critic_penalty`, on the recipe's LR
  schedule (`scale_learning_rates`). Removed the pinned
  `GradientPenalty(arm="b_cap", lazy_k=4)` critics (MoG AE/VAE, CIFAR particle
  AE/DDGAN, gym slider/particle fine-tunes, YuE2 and safe-fast 2D gates), the
  gym fine-tune `adv_posture` (`yue2` / `locked_shared`) and its
  `locked_shared.yaml`, the `--reg_arm`/`--r1_gamma`/`--loss_type`/`--gan_mode`
  menu of `examples/100gaussians.py`, the `reg_arm` key of the config-driven
  trainers (CIFAR DDGAN, denoising, 100-Gaussian runner, sparse; stripped from
  their launch configs; the finite-difference speed configs are deleted), and
  the hand-built Adam optimizers for G/E/prior/critics. Image-scale trainers
  keep a lazy penalty (`reg_every=4`; laziness changes only frequency and
  coefficient) and a full-budget network LR horizon as recipe fields. Frozen
  bundles under `reports/` and the `benchmarks/` research harnesses are
  unchanged.
- **Familiar GAN loop: the recipe's optimizers do the step-time work.** The
  loop is plain PyTorch — `d_loss = adv + penalty(D, real, fake)`, then
  `opt_d.zero_grad(); d_loss.backward(); opt_d.step()` and the same for
  `opt_g`. `recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))`
  returns `torch.optim.Adam` subclasses whose `step()` runs the spike guard,
  EMA-critic update and LR record (critic) and A2 latent damping /
  direct-particle response (generator). New role-named factories
  `recipe.make_critic_optimizer(D, ema_critic=...)` (additional critics) and
  `recipe.make_generator_optimizer(params, latent_table=None,
  direct_particles=None)`. `recipe.make_critic_penalty(opt_d, output=None,
  generator=None, collect_stats=False)` returns a callable penalty paired with
  that optimizer: `penalty(D, real, fake, *cond, **cond_kw)` returns a scalar,
  forwards conditioning to the critic and its EMA (roles of a shared module
  and wrappers such as the new `InputNoise` resolve to the matching EMA),
  takes the logits from a tuple's first element unless `output=` says
  otherwise, reads its step from the optimizer (no `step` argument) and
  exposes `last_stats`/`diagnostics()`. All regularization state (EMA critic,
  LR record, counters, histories) is in `optimizer.state_dict()` under
  `"regularizer"`, so the usual `torch.save({... "opt_g": opt_g.state_dict(),
  "opt_d": opt_d.state_dict()})` resumes exactly. `GANTrainer` checkpoints
  are schema 3 (schema 2 is upgraded on load); `trainer.penalty` is the recipe
  penalty and `trainer.critic` / `trainer.generator_regularizer` are gone.
  Removed: `make_critic_regularizer`, `make_generator_regularizer` and their
  `step`/`before_step`/`after_step`/`around` surface. Behavior note:
  `make_optimizers` now applies the recipe's spike guard (`d_guard_ratio`) and,
  for a learnable `ParticlePrior`, A2 damping (`latent_damping_max_rate`) even
  without a penalty; set both to 0 for plain Adam steps. GANTrainer,
  `examples/pytorch_loop.py` and the frozen-K3P parity tests are bit-identical.
- **Regularizers come from the recipe.** `recipe.make_critic_regularizer(D,
  opt_d, **penalty_kwargs)` (one call per critic optimizer) and
  `recipe.make_generator_regularizer(opt_g, latent_table=prior.z,
  direct_particles=None)` build whatever the current best formulation needs
  (K3P today) behind a formulation-agnostic interface: `penalty`,
  `ema_critic`, `before_step`/`after_step`/`step`, `diagnostics`,
  `state_dict`/`load_state_dict`. They replace `K3PCritic(recipe, ...)` and the
  `make_critic_anchor`, `make_critic_guard`, `make_latent_damping` and
  `make_direct_response` factories (removed). `K3PCritic`, the new
  `K3PGeneratorRegularizer` and the K3P primitives move to / stay in
  `particlegan.k3p` and are no longer exported from `particlegan`.
  `GANTrainer`, examples, experiments, `lib/` and docs use the factories;
  `trainer.generator_regularizer` is new. Math and checkpoints are unchanged.
- **K3P is the default.** `Recipe()`/`get_recipe()` now resolve to the qualified
  K3P config: `reg_arm="k3p"`, coefficient 1, κ 1, betas (0, .999), no particle
  spread, batch 2048, z_dim 2, plus new fields `network_lr_floor` (.01),
  `network_lr_horizon_cap` (1600), `reg_anchor_decay`, `d_guard_ratio`,
  `d_guard_min_steps`, `latent_damping_max_rate`, `direct_particle_betas` and
  the input/output noise schedules. New `learning_rate_scales(step, recipe)`.
- A per-critic-optimizer K3P bundle (now `recipe.make_critic_regularizer`)
  (trainer-allocated EMA critic with buffer averaging and side-effect-free
  forwards, penalty, spike guard, `state_dict`). Several critics use several
  bundles; one module with several roles passes a per-role `ema_critic`.
- `GANTrainer` trains K3P end to end (role-wise LR schedule, critic input and
  generator output noise from its own stream, spike guard, A2 latent damping)
  and checkpoints it (schema 2). Schema-1 GAN v3 checkpoints are rejected.
  `trainer.ema_D` is now a property of `trainer.critic`.
- Custom loops in `experiments/` and `lib/` use the critic bundle; hosts whose
  protocol requires `b_cap` pin it explicitly. Historical benchmarks resolve
  archived recipes through `benchmarks.gan_v3` (`GAN_V3_FIELDS`,
  `legacy_recipe`, `legacy_dict`), so their receipts are unchanged.
- `GradientPenalty()` now defaults to `arm="k3p"`; legacy direct callers pin
  `arm="b_cap"`. New `scale_learning_rates(step, recipe, optimizers,
  base_rates, prior)` sets G/D groups to the network schedule and prior groups
  to the prior one, so a custom loop's critic LR reaches the same floor K3P's
  blend weight uses. The custom loops, `examples/five_modes.py` and the
  README/API examples use it.
- Docs: new `docs/k3p.md`; GAN v3 docs marked superseded. The shipped
  `configs/100gaussians` and `configs/denoising` defaults follow the recipe.
- Add K3P as package components: `GradRegularizer(arm="k3p")` with
  `blend_weight()`, `after_critic_step()` and `state_dict()`, plus
  `CriticAnchor`, `CriticSpikeGuard`, `LatentRowDamping` and
  `DirectParticleResponse` in `particlegan.k3p`. One instance per critic, with
  no optimizer hooks or module globals. The caller allocates the EMA critic
  and the history buffers. With one critic, the result matches the frozen K3P
  mechanism bit for bit (`tests/test_k3p.py`). The recipe default does not
  change yet.
- `GANTrainer` and `examples/100gaussians.py` support `reg_arm="k3p"`: they
  call `after_critic_step` after each critic step and allocate the EMA critic.
  The trainer also averages the critic's buffers and saves both `ema_D` and
  the penalty state in its checkpoint.
- A k3p regularizer now raises if it is used on a second critic without an
  explicit `ema_critic=`, and `after_critic_step` accepts a tensor LR.

## 0.7.0 — 2026-09-24

- Add the strict 100-mode toy gate (`python -m benchmarks.toy100 run`); its
  default is the simpler shared recipe that passes all 22 toys. The README
  animation now shows that default converging on the 10×10 grid.

- Keep training control flow separate from recipes: construct
  `GANTrainer(recipe, G, D, ...)` explicitly. Remove `Recipe.make_trainer`;
  recipes retain hyperparameters and small component factories. The current
  winning hyperparameters remain unchanged.

- Restore named model-family selection with `get_recipe(name="gan", **overrides)`:
  GAN, MoG, DDGAN, AE-GAN, VAE-GAN and AE-DDGAN configurations share current
  optimizer, cap and spread defaults. Names select components without training
  control flow or legacy hyperparameters. Full saved recipe dictionaries still restore.
- Preserve historical leaderboard comparisons as benchmark inputs outside the
  installable package. The quickstart uses the winning batch-distance D directly.

- Add `particlegan.locked_shared.LOCKED_SHARED`, the demo RpGAN + `b_cap`
  stamp (cover 1.5, FM off, 12 particles, host critic). Builders are
  `make_gan_loss` and `make_b_cap`. This stamp is separate from the current
  recipe hyperparameters.

## 0.6.0 — 2026-09-24

- Require explicit keyword `sigma` in `MoGParticlePrior`; construction no longer
  searches nearest neighbors. The shared isotropic noise remains fixed in training.
- Add optional `calibrate_mog_sigma(centers, sigma_rel)` returning `(sigma, d0)`;
  retain exact even-count median and historical dtype rounding. Recipes explicitly
  calibrate their initialized centers unless `make_prior(sigma=...)` overrides them.
- Preserve legacy checkpoint centers, sigma, d0, read settings, samples and RNG
  behavior. Load with matching dimensions and `sigma=0`, then `load_state_dict`.
- Migrate fixed-noise integrations to `MoGParticlePrior(..., sigma=fixed_sigma)`
  and remove post-construction sigma overwrites. Replace `calibrate()` with the
  standalone helper only when spacing-based calibration is intended.

## 0.5.0 — 2026-09-17

- Add public `particle_ae`, `particle_vae` and `ParticleEncoding`, plus
  `get_recipe("ae_gan")`, `get_recipe("vae_gan")` and `get_recipe("ae_ddgan")`.
  Caller-owned encoders select learned MoG particles for reconstruction.
- Default VAE uses one selected particle with prior-matching fixed-sigma noise:
  its joint KL is constant, so no KL penalty is needed in training.
  Reconstruction helpers never add KL. Gaussian negative ELBO reporting and
  the soft categorical posterior are explicit opt-ins; hard routing has a
  biased straight-through gradient. The genuine VAE evidence is toy-only.
- Add `recipe.encode(...)` and optional `encoder=E` to `make_optimizers` with
  shared-parameter deduplication. Existing recipes retain their defaults;
  networks, loops, loss composition, EMA and optimizers remain caller-owned.
- Document AE-DDGAN one-step reconstruction, inference and numerical variation
  audits. Matched CIFAR32 at 10k updates: FID50k 19.483 direct GAN, 20.054
  AE-GAN, 43.233 AE-DDGAN and 49.475 DDGAN. These single-trajectory results
  do not establish universal superiority or image VAE performance.
- Include queued toy/image experiments, configs, portable leaderboards,
  provenance and tests. Document narrower-than-real modes and late instability.

See the [particle autoencoder guide](docs/particle-autoencoders.md) for objectives,
examples, public contracts and evidence. No additional core dependencies.

## 0.4.0 — 2026-09-17

- Add `get_recipe("ddgan_mog")`: DDGAN with class-only UCD, 400 MoG components,
  z_dim=4, sigma_rel=1/40, standardized reads, 100,000 updates, a constant learning
  rate, prior LR multiplier 100 (0.06), and prior Adam betas (0.5, 0.999).
  Existing GAN, MoG and DDGAN presets retain their defaults; keyword overrides
  remain supported.
- Support MoG latent priors in `train_denoising` and checkpoint probes, including
  raw-mean regularization and separate prior optimizer settings. Add
  `generator_hidden` to vary generator width independently of the discriminator.
- Include matched 14k/100k GAN/DDGAN × atoms/MoG studies, frozen-noise probes,
  configs, metrics and readouts. At 100k, DDGAN+MoG reaches 79.57% joint HQ and
  100 modes with core width ratio .889; the one-shot models retain higher HQ
  but cover 77 modes. These are single-seed findings with a small generator,
  not a universal quality guarantee or an isolated capacity result.

The new recipe supplies package hyperparameters from the 100k study. Networks,
data, training/sampling loops and EMA remain caller-owned; the study used G
width 32 and D width 128. See the
[100k readout](reports/denoising-toy/mog_capacity_100k/READOUT.md).

## 0.3.0

- Add public `MoGParticlePrior`: a uniform mixture with learned means and a
  shared, fixed Gaussian sigma calibrated from initial nearest-neighbor spacing.
  Defaults to **400 components, z_dim=4, sigma_rel=1/40, standardized reads**.
- Add `get_recipe("mog")`, the selected **400-component, 28k-step** recipe:
  prior LR 0.06, prior Adam betas (0.5, 0.999), and the existing GAN defaults.
  Recipe factories support selecting the prior and setting prior betas separately.
- Add `configs/mog/default.toml` to run that experiment with the existing
  100-Gaussian trainer. Existing examples and the `ParticlePrior` atoms API retain
  their behavior.
- MoG supports explicit sampling generators, fixed epsilon snapshots, noisy
  module forward calls for DDP, raw-center regularization, EMA, and state-dict
  restoration of read configuration and calibrated noise. Legacy experimental
  checkpoints remain loadable with their original standardization setting.
- Keep the core dependency on PyTorch alone. The optional `mog` extra installs
  SciPy for faster calibration of large low-dimensional tables; a memory-bounded
  exact Torch fallback is available without it.

The selected compact model passed the frozen C0 envelope on 100 Gaussians at
28k steps: HQ/real 0.99953, width/real 0.92636, KL 0.02888. It uses 50 times fewer
components and four times the training steps of the original baseline. This is
a single-seed configuration result, not a claim of universal superiority.
See [the full experiment report](results/mog/COMPONENT_SCALE.md).
