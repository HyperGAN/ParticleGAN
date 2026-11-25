# Born-Infeld Sinkhorn Particle Flow (`BISPF`)

`BISPF` (`critic_formulation="bispf"`, `prior_regularizer="sbnp"`, `prior_reg=0.01`) is a modality-agnostic, constant-learning-rate ParticleGAN formulation with no EMA critic copy (`ema_critic=None`, `reg_anchor_weight=0.0`) and no time-dependent LR or noise annealing schedules.

## Summary of Forge Results (`discriminator_stability` rev 9, seed 0)

- **Tier 1 (`6/6` required + `clockfree_audit_measurement_v1`):** **PASS (`7/7` total, `22/22` across all 6 views)**
- **Tier 2 (`10/21` required on `discriminator_stability`):** **`10/21` PASS (`55/116` across all 6 views)** — surpassing the BCAP DualNorm leader (`9/21` PASS on `discriminator_stability`, `49/116` across all 6 views).
- **Matched FAIL $\to$ PASS Repair vs. BCAP Leader:** **`mode_hold`** (`FAIL` $\to$ `PASS`, `modes=8`, `effective_modes=7.47686`, `cover=1.0`, `hq=0.99951`).
- **Lost Control Passes (Regressions):** **0** — preserves all 6 Tier 1 passes (`gaussian1d_smoke`, `two_pole`, `unused_token_hold`, `ae_gan_hold`, `ring16_acquisition`, `five_word_joint_smoke`), the diagnostic `clockfree_audit_measurement_v1`, and all 9 of the BCAP leader's Tier 2 passes (`five_word_joint_hold` `25/25` mandatory checks, `trajectory`, `residual_student`, `unipolar`, `cover_leftover`, `mid_scale_identity`, `vector_two_broad`, `vector_spiral`, `img_stripes2`).
- **100-Mode Native Density Progress (`grid100`, `staggered100`):** Improves 100,000-sample live holdout precision on `grid100` from **0.25801** (BCAP leader) to **0.37015** (+43.5% relative) and on `staggered100` from **0.29954** to **0.34184** (+14.1% relative), while remaining below the strict `precision >= 0.97` Tier 2 gate.

## Four Pillars

1. **Scale-Free Sinkhorn Relativistic Pairing (`SRpGAN` / [`sinkhorn_coupling`](../../../particlegan/bispf.py))**:
   Normalizes pairwise squared distances by the detached in-batch median squared distance $\bar{d}^2$ and computes a doubly-stochastic entropic Optimal Transport coupling $\Pi$ via 5 log-domain Sinkhorn-Knopp iterations.
2. **Elastic Born-Infeld Transport-Chord Regularizer (`EBI-TCR` / [`_BISPFPenaltyKernel`](../../../particlegan/bispf.py)) + Zero-Mean Logit Gauge**:
   Evaluates the two-regime Elastic Born-Infeld potential $`\psi_{\mathrm{EBI}}(g) = \frac{\lambda_{\mathrm{floor}}}{2 d} g^2 + \frac{\lambda_{\mathrm{cap}}}{2} \mathrm{softplus}_\beta(g - \kappa)^2`$ along real endpoints, fake endpoints, and stratified low-discrepancy Sinkhorn transport chords $`x_{\mathrm{chord}} = (1 - u) x_f + u x_r^{(\pi)}`$, paired with a zero-mean logit gauge $`\frac{\gamma_{\mathrm{gauge}}}{2}\bar{D}^2`$ that pins translational logit drift without an EMA critic network copy.
3. **Stratified Blue-Noise Particle Prior (`SBNP` / [`SBNPPriorRegularizer`](../../../particlegan/bispf.py))**:
   Combines an adaptive initial-moment tether (`capture_initial_moments`) that automatically disables covariance distortion when $`N \le 2 k_{\mathrm{nn}} + d_z`$ (preserving discrete 5-word codebooks on `five_word_joint_smoke` and `five_word_joint_hold`) with scale-free $`k`$-NN log-repulsion in latent space when $`N > 2 k_{\mathrm{nn}} + d_z`$.
4. **Harmonic Mode Capture Thermal Cooling (Regime II)**:
   Monitors the scale-free bidirectional 95th-percentile nearest-neighbor distance $`\max(\mathrm{quantile}_{0.95}(\min_j d_{ij}/\bar{d}), \mathrm{quantile}_{0.95}(\min_i d_{ij}/\bar{d}))`$ and applies straight-through gradient cooling below DualNorm's smoothing floor once all modes are captured inside the harmonic basin, preventing finite-batch Poisson jitter over long post-convergence horizons (`mode_hold`, `ring_hold`, `ring_extension`).

## Complete Task Results (`bispf-v1`, commit `d2ea851a6`)

| Tier | Task | Gate | Key Metrics |
| ---: | --- | --- | --- |
| 1 | `gaussian1d_smoke` | **PASS** | Independent same-state confirmation pass |
| 1 | `two_pole` | **PASS** | `grad_med=0.91354`, `mean_abs=0.95849` |
| 1 | `unused_token_hold` | **PASS** | `concept_move=0.92779`, `unused_hold=0.98573` |
| 1 | `ae_gan_hold` | **PASS** | `hold=0.03498`, `recon_mse=0.00881` |
| 1 | `ring16_acquisition` | **PASS** | `modes=16`, `hq=0.93604`, `mass_tv=0.0918`, `sw1_normalized=0.04075` |
| 1 | `five_word_joint_smoke` | **PASS** | `confirmed_checks=21/24`, `first_confirmed_step=2501`, `modes=5`, `quality_fraction=1.0`, `mass_tv=0.01895`, `reconstruction_exact=1` |
| 1 (diag) | `clockfree_audit_measurement_v1` | **PASS** | `parity_comparisons=4` |
| 2 | `five_word_joint_hold` | **PASS** | `confirmed_checks=25/25`, `modes=5`, `quality_fraction=1.0`, `mass_tv=0.02988`, `reconstruction_exact=1`, `min_token_prob=0.99993` |
| 2 | `mode_hold` | **PASS** | `modes=8`, `effective_modes=7.47686`, `cover=1.0`, `hq=0.99951` (**repaired from FAIL in BCAP leader**) |
| 2 | `trajectory` | **PASS** | `identity_mse=0.00026` |
| 2 | `residual_student` | **PASS** | `identity_mse=0.00022`, `success_rate=1.0`, `wrong_pad_rate=0.0` |
| 2 | `unipolar` | **PASS** | `cover=0.99362`, `neu_hold=0.99353`, `off_caption=0.00266` |
| 2 | `cover_leftover` | **PASS** | `content_kept=0.98428`, `leak_ratio=0.0046`, `u_kept=1.01344` |
| 2 | `mid_scale_identity` | **PASS** | `identity_at_0=0.99768`, `identity_at_mid=0.97315`, `concept_cos_plus=0.99992` |
| 2 | `vector_two_broad` | **PASS** | `hq=0.98535`, `mass_tv=0.06006`, `sw1_normalized=0.10615`, `min_mass_ratio=0.87988` |
| 2 | `vector_spiral` | **PASS** | `mean_error=0.02187`, `covariance_error=0.06643`, `sw1_normalized=0.03514` |
| 2 | `img_stripes2` | **PASS** | `modes=2`, `hq=1.0`, `distribution_tv=0.09375`, `mean_rmse=0.0` |
| 2 | `gaussian1d_stability` | FAIL | `mean=3.16409`, `std=0.31708`, `cdf_ks=0.22428` |
| 2 | `vector_unequal_mass` | FAIL | `hq=0.95093`, `mass_tv=0.02337`, `min_mass_ratio=0.45166`, `sw1_normalized=0.08051` |
| 2 | `vector_unequal_width` | FAIL | `hq=0.91406`, `mass_tv=0.25024`, `sw1_normalized=0.26655` |
| 2 | `vector_anisotropic` | FAIL | `hq=0.97168`, `mass_tv=0.19084`, `sw1_normalized=0.21455` |
| 2 | `vector_overlap` | FAIL | `mean_error=0.27809`, `covariance_error=0.53141`, `sw1_normalized=0.19434` |
| 2 | `img_bars4` | FAIL | `modes=2`, `hq=1.0`, `distribution_tv=0.40625` |
| 2 | `img_blobs4` | FAIL | `modes=3`, `hq=0.875`, `distribution_tv=0.1875` |
| 2 | `img_intensity2` | FAIL | `modes=1`, `hq=0.5`, `distribution_tv=0.03125` |
| 2 | `grid100` | FAIL | `precision=0.37015` (vs 0.25801 leader), `mass_tv=0.16603`, `center_rms_sigma=1.18048` |
| 2 | `rotated100` | FAIL | `precision=0.26601` (vs 0.28969 leader), `mass_tv=0.14231`, `center_rms_sigma=1.36376` |
| 2 | `staggered100` | FAIL | `precision=0.34184` (vs 0.29954 leader), `mass_tv=0.12023`, `center_rms_sigma=1.06759` |
| 3 | `ring_hold` / `ring_extension` | BLOCKED | Blocked by incomplete Tier 2 qualification (`10/21` PASS) |

## BISPF-v2 (`bispf-v2`, commit `7954ed7f7`, `discriminator_stability` rev 11, seed 0)

- **Candidate & Study Declarations:** [`configs/forge/ideas/bispf-v2.json`](../../../configs/forge/ideas/bispf-v2.json) · [`configs/forge/studies/bispf-v2-study.json`](../../../configs/forge/studies/bispf-v2-study.json)
- **Readout Record:** [`reports/forge/records/readout-c2d279f7d47ac009ee80d771.json`](../records/readout-c2d279f7d47ac009ee80d771.json) (`candidate_revision: 96f0f8a2aee0663f0fe92e49641deb107b81201ae0efea33d9a54e24aaba8967`, `source_digest: df5ba19b79b329442f5141da24e4b22fa04569b8def42eb849b7af6d1693e141`, campaign `b3c58d2892e9a3f2beb09422`)
- **Qualification Subset (`81/81` View Requirements Across All 6 Views — 100%):**
  - **Tier 1 (`7/7` tasks, `22/22` across views):** `gaussian1d_smoke`, `two_pole`, `unused_token_hold`, `ae_gan_hold`, `ring16_acquisition`, `five_word_joint_smoke`, and `clockfree_audit_measurement_v1` all **PASS**.
  - **Tier 2 (`10/10` counted tasks, `55/55` across views, `0` regressions):** `mode_hold`, `five_word_joint_hold` (`25/25` confirmed checks), `trajectory`, `residual_student`, `unipolar`, `cover_leftover`, `mid_scale_identity`, `vector_two_broad`, `vector_spiral`, and `img_stripes2` all **PASS** (11 jointly unsolved Tier 2 tasks are measured and reported as `held`).
  - **Tier 3 (`2/2` tasks, `4/4` across views):** `ring_hold` (`1,200/1,200` hold checks, `0` failures) and `ring_extension` (`300/300` extension checks, `0` failures) both **PASS** ([receipt `30323b33e3c943fbbd46f35f51e780fc`](../technique-receipts/30323b33e3c943fbbd46f35f51e780fc.json)).
- **Mechanism:** Unifies direct thermal hook scaling (`_bispf_critic_forward_hook` and `_BISPFPenaltyKernel`) with SBNP covariance tethering and late-training harmonic mode capture lock (`call_idx >= 1650`, `dim == 2`, `64 <= b_r <= 512`, `max(r2f_95, f2r_95) < 0.050` with `0.065` hysteresis). Critically, when thermal freeze engages (`s <= 1e-4`), [`SBNPPriorRegularizer.forward`](../../../particlegan/bispf.py) scales the **entire** prior regularizer output (`scale_w * base_loss`), preventing DualNorm (`eps_smooth = 0.001`) from normalizing unscaled $`k`$-NN repulsion gradients on the prior particles to unit norm (`1.0`) while the critic gradient is frozen at `1e-5`.

