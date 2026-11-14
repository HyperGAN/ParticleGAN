<!-- Generated Forge problem list -->
# Unresolved training problems

[Full-suite leaderboard](technique-inventory.md) · [All test/protocol results](test-performance.md)

This page explains the outstanding tests, including the held failures. **Held means excluded from qualification, not solved:** these failures are included in the leaderboard's full-suite comparison. Passing one problem does not establish that the same configuration passes the whole suite.

Selected results below use one recorded recipe/source/runtime cohort per family. Research, diagnostic and historical configurations are named separately, with links to their original evidence. An oracle or hand-built solution demonstrates feasibility, not successful training. Not measured, blocked, timeout and error are never relabelled FAIL.

Passer summaries reflect the evidence audit dated **2026-10-15**. Selected-result tables are regenerated from the registered publication.

| Problem | What it tests | Has any configuration passed? |
| --- | --- | --- |
| [Keep a bell curve stable, then learn a moved target](problems.md#gaussian1d_stability) | After learning a simple bell curve, keep its centre, width and distribution correct while training continues; then track a shifted target and stay there. | Selected recorded PASS: EMPF, RHBI. |
| [Preserve rare clusters](problems.md#vector_unequal_mass) | Learn several blobs with different probabilities, including a rare 2% blob, without losing its population or collapsing its shape. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Learn narrow and broad clusters together](problems.md#vector_unequal_width) | Give four equally likely blobs different widths while preserving both their shapes and their equal probabilities. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Learn stretched and tilted clusters](problems.md#vector_anisotropic) | Match the populations and shapes of three equally likely blobs stretched horizontally, vertically and diagonally. | Selected recorded PASS: RHBI. |
| [Settle on an overlapping distribution](problems.md#vector_overlap) | Match the observable centre, spread and distribution of two overlapping blobs; no hidden component labels are required. | Selected recorded PASS: RHBI. |
| [Generate all four bar images](problems.md#img_bars4) | Generate each of four simple bar pictures, rather than producing only a few convincing examples. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Generate all four image corners](problems.md#img_blobs4) | Generate a small bright square at each of four corners, with enough clean examples of every corner. | Selected recorded PASS: RHBI. |
| [Generate both brightness levels accurately](problems.md#img_intensity2) | Generate a square at either of two target brightnesses, with both levels represented and little brightness error. | Selected recorded PASS: RHBI. |
| [A grid of 100 clusters](problems.md#grid100) | Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [A rotated grid of 100 clusters](problems.md#rotated100) | Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [A staggered grid of 100 clusters](problems.md#staggered100) | Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Keep the 100-cluster grid correct for longer](problems.md#grid100_14k) | Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Keep the 100-cluster rotated grid correct for longer](problems.md#rotated100_14k) | Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Keep the 100-cluster staggered grid correct for longer](problems.md#staggered100_14k) | Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Recover quickly after the target moves](problems.md#target_shift_recovery) | Learn and hold a ring of blobs, then track a sudden sideways shift within 400 updates and stay correct; a frozen model must fail to show the change is real. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Keep every learned mode alive](problems.md#mode_hold) | Keep generating every target mode during continued training, rather than losing modes after initially finding them. | Selected recorded PASS: BISPF, EMPF, RHBI. |
| [Keep a learned ring stable for a long run](problems.md#ring_extension) | Acquire all eight ring blobs, hold them for 1,200 consecutive checks, then survive 300 additional checks on the same uninterrupted run. | Selected recorded PASS: BISPF. |
| [Learn sixteen distinct clusters from scratch](problems.md#ring16_acquisition) | Learn all sixteen blobs on a ring with balanced populations and real within-blob spread, rather than sixteen collapsed points. | Selected recorded PASS: BCAP, BISPF, EMPF, RHBI. |
| [Learning must not depend on external clock labels](problems.md#clockfree_audit_measurement_v1) | Check that the same learning state and inputs produce the same update despite changed step labels, declared horizon, evaluation cadence or a save/restart. | Selected recorded PASS: BCAP, BISPF, EMPF, RHBI, Atlas. |
| [Escape a tanh saturation rail after the target shifts to zero](problems.md#zero_tanh_rail_escape) | First converge near the tanh saturation rail at (+0.90, -0.90) where tanh' is attenuated (~0.19), then escape the rail within 400 updates when the target shifts to (0.0, 0.0) at update 1,001 and stay settled there. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Survive a 16-layer unnormalized MLP generator on the 100-cluster grid](problems.md#grid100_mlp16) | Learn all 100 equal-weight Gaussian modes on `grid100` using a 16-hidden-layer 128-wide LeakyReLU(0.2) MLP generator without residual skips or normalization layers. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Learn the 100-cluster grid with a 16-layer residual MLP and bounded tanh head](problems.md#grid100_res16_tanh) | Recover all 100 Gaussian modes on `grid100` using a 16-layer identity-skip residual MLP generator and a bounded `S=5.25` scaled `tanh` output head without saturating the `\|x\| > 0.99 * 5.25` rails. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Recover an 8D lifted 100-mode manifold from incomplete block-masked data (MisGAN + ppost)](problems.md#misgan_block_recovery) | Train a 2-pair incomplete-data MisGAN (`(G_x, D_x)` + `(G_m, D_m)` with detached mask gradient in `L_x`) observing only masked 8D data `(x * m, m)` under structured block dropout, then impute held-out rows via native Bayesian particle-posterior (`ppost`) resampling from `G_x`. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Learn a multimodal neural imputer on ambiguous single-sensor block masks without ppost](problems.md#misgan_block_neural_imputer) | Train a 3-pair MisGAN (`(G_x, D_x)` + `(G_m, D_m)` + `(G_i, D_i)`) and grade the neural imputer `G_i([x * m, m, z_i])` directly on the 16 held-out ambiguous single-sensor queries without `ppost` particle resampling. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Learn a continuous 16-mode ring joint Encoder-Generator-Critic BiGAN](problems.md#ring16_bigan_continuous) | Train a continuous 2D multimodal joint BiGAN `(G(z), E(x), D(x, z))` on the 16-Gaussian ring (`MoGParticlePrior(sigma=0.1)`), simultaneously satisfying forward 16-mode generation and paired round-trip reconstruction `x -> G(E(x))`. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Separate direct sample particles across two symmetric poles](problems.md#two_pole) | Drive a generator-free direct particle cloud from zero to two symmetric poles (`mean_abs >= 0.30`) and hold them steady with bounded critic gradient (`grad_med <= 1.0`) across the final five checks. | Selected recorded PASS: BCAP, BISPF, EMPF, RHBI, GAN v3 release 0.7, K3P, KA2, R1/R2. |
| [Acquire five discrete words and paired BiGAN reconstructions in 20,001 joint updates](problems.md#five_word_joint_smoke) | Train a joint generator-encoder-prior-critic BiGAN over five five-letter words for 20,001 updates, acquiring all five clean word modes (`modes == 5`, `quality_fraction >= 0.95`, `mass_tv <= 0.10`) and exact high-confidence paired reconstructions (`reconstruction_exact == 1`, `minimum_reconstruction_token_probability >= 0.90`) at an independently confirmed scheduled state. | Selected recorded PASS: BCAP, BISPF, EMPF, RHBI, GAN v3 release 0.7, K3P, KA2, R1/R2. |
| [Reconstruct four noisy corner Gaussians and generate balanced independent modes (AE-GAN)](problems.md#model_ae_gan_multimodal) | Train a public-API multimodal AE-GAN to reconstruct four equally weighted corner Gaussians (`reconstruction_mse <= 0.025`, `reconstruction_mode_agreement >= 0.98`) while independently generating all four modes with balanced masses (`mass_tv <= 0.15`) and well-conditioned within-mode covariances (`0.10 <= eigen_ratio <= 4.0`) across the final five checks. | Selected recorded PASS: EMPF, RHBI. |
| [Learn four-step conditional reverse transitions and ancestral generation (DDGAN)](problems.md#model_ddgan_gaussian_chain) | Train a four-step public-API DDGAN on a 2D Gaussian forward diffusion chain so every reverse conditional transition (`t1..t4`, `q0`/`q1`) and the final ancestral marginal satisfy `projection_cdf <= 0.12`, `normalized_mean <= 0.25`, and covariance eigenvalues in `[0.65, 1.40]` across the last five checks. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Pair particle AE reconstruction with four-step DDGAN reverse transitions and ancestral generation](problems.md#model_ae_ddgan_gaussian_chain) | Reconstruct held-out inputs through a learned particle autoencoder and four public DDGAN reverse steps (`paired_nmse <= 0.10`, `paired_worst_axis_nrmse <= 0.25`, `paired_shuffled_gain >= 3.0`) while simultaneously passing all conditional reverse-transition and independent ancestral-chain bounds. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Learn a Gaussian with 8-bit MXFP8 training arithmetic](problems.md#precision_gaussian_mxfp8) | Train from the shared deterministic initialization using Blackwell-format MXFP8 software-reference forward and backward GEMM operands; match the full scalar Gaussian law at five final checks. | Selected recorded PASS: RHBI. |
| [Learn a Gaussian with 4-bit NVFP4 training arithmetic](problems.md#precision_gaussian_nvfp4) | Train from the shared deterministic initialization using Blackwell-format NVFP4 software-reference forward and backward GEMM operands; match the full scalar Gaussian law at five final checks. | Selected recorded PASS: RHBI. |
| [Hold all five discrete words and exact BiGAN inversion across 4,000 continuation updates](problems.md#five_word_joint_hold) | Restore the candidate's own confirmed five-word acquisition checkpoint in a fresh process and pass every generation and exact encoder-inversion bound across all 24 scheduled checks over 4,000 additional updates. | Selected recorded PASS: BCAP, BISPF, RHBI. |
| [Interpolate unseen latent k-NN chords along a 270-degree circular arc from 12 zero-sigma particles without MoG](problems.md#interp_sparse_arc) | Train 12 discrete zero-sigma `ParticlePrior` locations and an MLP generator on a 2D circular arc (`R = 2.0`, `sigma = 0.03`) under protocol seed 0 so both endpoints (`endpoint_hq >= 0.85`) and unseen latent k-NN chords (`interp_hq >= 0.80`, `interp_arc_bins >= 22`, `midpoint_novelty >= 0.20`) stay on the curved manifold across five final live checks. | No selected recorded PASS; see exact outcomes and dated diagnostic evidence below. |
| [Translate a sharp 2x2 patch smoothly across unseen latent chords in 8x8 images from 16 zero-sigma particles](problems.md#interp_img_patch_shift) | Train 16 zero-sigma `ParticlePrior` codes (`z_dim = 4`) and a `residual_upsample` convolutional generator on continuous single-patch 8x8 images so unseen latent k-NN chords render single sharp shifted patches (`interp_rmse <= 0.10`, `interp_peak >= 0.78`, `midpoint_novelty >= 0.18`, `quadrant_coverage == 4`) rather than 0.5-peak pixel double exposures. | Selected recorded PASS: RHBI. |
| [Interpolate unseen 2D latent triangle barycenters across a curved 3D saddle sheet from 24 zero-sigma particles](problems.md#interp_saddle_barycentric) | Train 24 zero-sigma `ParticlePrior` locations (`z_dim = 2`) and a 3D-output MLP generator on $w = 0.8(x^2 - y^2) + 0.9\cos(0.75\pi x)\cos(0.75\pi y)$ so unseen 2D latent k-NN triangle barycenters stay on the curved surface (`barycenter_surface_mae <= 0.09`, `barycenter_hq >= 0.78`, `xy_bins_covered >= 12`, `barycenter_novelty >= 0.18`) across five final live checks. | Selected recorded PASS: RHBI. |

<a name="gaussian1d_stability"></a>

## Keep a bell curve stable, then learn a moved target

Test: `gaussian1d_stability` · Required for qualification and included in full-suite results.

**What it tests.** After learning a simple bell curve, keep its centre, width and distribution correct while training continues; then track a shifted target and stay there.

**Historical diagnosis (2026-10-15).** BCAP and BISPF keep oscillating instead of settling. BISPF passes only 5 of 72 stationary checks, misses deadline re-acquisition and passes 0 of 24 shifted-hold checks. A one-time smoke pass does not establish stability.

**Historical passer evidence.** Yes, individual research configurations; no selected recipe passes the complete test.

- `kinetic-local-nsgda-cross-v1`, `kinetic-local-dualnorm-g-history-v1` and `kinetic-local-dualnorm-g-history-bank8-v-v1` pass, but fail ring16 and mode hold.
- Cell-gradient-signal configurations and `cell-quota-minimum-change-forward-kl-g-v1` also pass. The latter passes mode hold too, but its later ring16 probe fails and it cannot run two-pole or word smoke.
- A hand-built network can draw the target; that is a capacity witness, not a trained full-protocol PASS.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-gaussian1d_stability) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-gaussian1d_stability) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-gaussian1d_stability) |

**Recommendation from the dated diagnosis.** Find an update rule that settles when correct but responds after a shift, then verify the same recipe on ring16 and mode hold. Existing trade-offs do not prove the problem impossible.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#gaussian1d_stability).


<a name="vector_unequal_mass"></a>

## Preserve rare clusters

Test: `vector_unequal_mass` · Held for qualification; included in full-suite results.

**What it tests.** Learn several blobs with different probabilities, including a rare 2% blob, without losing its population or collapsing its shape.

**Historical diagnosis (2026-10-15).** The rare blob receives little training signal. BISPF preserves its share but flattens its shape: minimum eigen ratio 0.004 against the required 0.15. BCAP also collapses it.

**Historical passer evidence.** Yes, transport and cell-moment research configurations; BCAP and BISPF fail.

- `kinetic_transport_local_v2` and the two kinetic g-history reference8 variants pass; they fail unequal-width and mode hold.
- Round-5 direction blend plus local-v2 passes, but fails Gaussian stability and unequal-width and was not measured on most of the required suite.
- Cell-moment aggregate recipes pass all four vector tests; they lose mode hold or ring16. True-target sampling passes 200/200 reference runs, which validates the gate but is not a training win.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_mass) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-vector_unequal_mass) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_mass) |

**Recommendation from the dated diagnosis.** Preserve rare-cluster coverage and shape while checking unequal-width, ring16 and mode hold with one unchanged recipe.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_unequal_mass).


<a name="vector_unequal_width"></a>

## Learn narrow and broad clusters together

Test: `vector_unequal_width` · Held for qualification; included in full-suite results.

**What it tests.** Give four equally likely blobs different widths while preserving both their shapes and their equal probabilities.

**Historical diagnosis (2026-10-15).** BISPF gets the shares wrong (mass TV 0.25) even where shapes pass. BCAP has both share and spread errors. A few strays can dominate a narrow blob.

**Historical passer evidence.** Yes, balanced assignment and cell-moment research configurations.

- Round-5 balanced coordinate assignment passes with 14 terminal checks and spread error 0.286, but loses unequal-mass, Gaussian stability and native 100-blob quality; much of the required suite is untried.
- Cell-moment aggregate and per-kernel recipes pass but fail other required tests. G-angle has a historical-source pass, not a current selected pass.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_unequal_width) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-vector_unequal_width) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_unequal_width) |

**Recommendation from the dated diagnosis.** Test the balanced-assignment reference on the missing suite and solve its rare-cluster regression before claiming a global improvement.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_unequal_width).


<a name="vector_anisotropic"></a>

## Learn stretched and tilted clusters

Test: `vector_anisotropic` · Held for qualification; included in full-suite results.

**What it tests.** Match the populations and shapes of three equally likely blobs stretched horizontally, vertically and diagonally.

**Historical diagnosis (2026-10-15).** Both leading recipes allocate the wrong amounts to the blobs. BISPF has mass TV 0.191 against 0.15 and covariance error 2.0.

**Historical passer evidence.** Yes: K3P, KA2, R1/R2 in separate diagnostics, plus research configurations.

- K3P, KA2 and R1/R2 pass in the attempt index, separately from qualifying selected-cohort results. They fail mode hold, ring16 and the clock measurement in the recorded comparison.
- Nine additional indexed research recipes and four cell-moment aggregates pass. Recorded passers with mode-hold measurements fail it.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_anisotropic) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-vector_anisotropic) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_anisotropic) |

**Recommendation from the dated diagnosis.** Retain the successful mass allocation while preserving mode hold and basic acquisition.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_anisotropic).


<a name="vector_overlap"></a>

## Settle on an overlapping distribution

Test: `vector_overlap` · Held for qualification; included in full-suite results.

**What it tests.** Match the observable centre, spread and distribution of two overlapping blobs; no hidden component labels are required.

**Historical diagnosis (2026-10-15).** BCAP reaches the target temporarily and drifts away. BISPF ends with centre error 0.28, covariance error 0.53 and distribution distance 0.19; it misses the sustained ending.

**Historical passer evidence.** Yes: K3P in separate diagnostics and multiple research recipes.

- K3P and ten other indexed configurations pass under the recorded declaration; all fail mode hold.
- Four cell-moment aggregates pass; three lose mode hold and the Bures variant loses ring16.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-vector_overlap) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-vector_overlap) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-vector_overlap) |

**Recommendation from the dated diagnosis.** Add stability without losing multimodal acquisition or mode hold. A good endpoint alone does not meet the sustained gate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#vector_overlap).


<a name="img_bars4"></a>

## Generate all four bar images

Test: `img_bars4` · Held for qualification; included in full-suite results.

**What it tests.** Generate each of four simple bar pictures, rather than producing only a few convincing examples.

**Historical diagnosis (2026-10-15).** BCAP and BISPF produce clean pictures but only 2 of 4 bars. Some transport recipes reach more bars but produce too many blurry pictures.

**Historical passer evidence.** Yes: R1/R2 and GAN v3 in separate diagnostic measurements.

- R1/R2 and GAN v3 pass this declaration, but fail ring16, the clock measurement and the required stripes test.
- Older historical recipes pass bars and stripes together under different contracts; they cannot fill today's cells.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_bars4) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-img_bars4) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-img_bars4) |

**Recommendation from the dated diagnosis.** Preserve all four clean bar modes while passing stripes and the acquisition tests with the same recipe.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#img_bars4).


<a name="img_blobs4"></a>

## Generate all four image corners

Test: `img_blobs4` · Held for qualification; included in full-suite results.

**What it tests.** Generate a small bright square at each of four corners, with enough clean examples of every corner.

**Historical diagnosis (2026-10-15).** BISPF reaches 3 corners and 28/32 clean pictures; the gate requires all 4 and at least 29/32. BCAP reaches 2 corners. A near miss or four passing checks is not a five-check sustained pass.

**Historical passer evidence.** Yes: R1/R2 and kinetic research configurations.

- R1/R2 passes in separate diagnostic measurements with an 18-check streak, but loses ring16 and the clock measurement.
- `kinetic_transport_local_v2` passes but fails two-pole and unused-token hold. Four kinetic history variants pass but are blocked on basic behavioural tests.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_blobs4) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-img_blobs4) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-img_blobs4) |

**Recommendation from the dated diagnosis.** Keep all four corners clean and sustained without breaking basic learning and identity preservation.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#img_blobs4).


<a name="img_intensity2"></a>

## Generate both brightness levels accurately

Test: `img_intensity2` · Held for qualification; included in full-suite results.

**What it tests.** Generate a square at either of two target brightnesses, with both levels represented and little brightness error.

**Historical diagnosis (2026-10-15).** BISPF collapses to one brightness level. BCAP produces both but 10/32 pictures have inaccurate brightness.

**Historical passer evidence.** Yes: GAN v3, R1/R2 and kinetic transport in separate measurements.

- GAN v3, R1/R2 and `kinetic_transport_local_v2` pass the recorded declaration, but fail required Tier 1 tests.
- A historical wider-network pass belongs to `img_intensity2_residual16`, a different task variant.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-img_intensity2) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-img_intensity2) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-img_intensity2) |

**Recommendation from the dated diagnosis.** Learn accurate, balanced brightness modes while retaining the other image and acquisition passes.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#img_intensity2).


<a name="grid100"></a>

## A grid of 100 clusters

Test: `grid100` · Required for qualification and included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**Historical diagnosis (2026-10-15).** BCAP/BISPF put only 24–37% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 54%, still below the gate.

**Historical passer evidence.** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-grid100) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-grid100) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100) |

**Recommendation from the dated diagnosis.** Improve both precise placement and within-cluster shape, then meet the sustained checks and independent holdout. Longer training alone has not solved it.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#grid100).


<a name="rotated100"></a>

## A rotated grid of 100 clusters

Test: `rotated100` · Required for qualification and included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**Historical diagnosis (2026-10-15).** BCAP/BISPF put only 26–27% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 51%, still below the gate.

**Historical passer evidence.** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-rotated100) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100) |

**Recommendation from the dated diagnosis.** Improve both precise placement and within-cluster shape, then meet the sustained checks and independent holdout. Longer training alone has not solved it.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#rotated100).


<a name="staggered100"></a>

## A staggered grid of 100 clusters

Test: `staggered100` · Required for qualification and included in full-suite results.

**What it tests.** Learn all 100 small blobs with correct locations, populations and widths, scoring the current model's clean outputs without adding evaluation noise.

**Historical diagnosis (2026-10-15).** BCAP/BISPF put only 30–34% of samples on target, against 97% required. Some blobs are thin or nearly empty; the best audited 7,000-update result reaches 56%, still below the gate.

**Historical passer evidence.** No full PASS under the current clean-sampling declaration in the audited evidence.

- Historical K3P and other recipes have PASS receipts under earlier contracts, including different priors, averaged weights or evaluation noise. Those are separate problems and do not beat this declaration.
- True-target samplers demonstrate feasibility. Longer BISPF training improves precision initially but plateaus below the gate.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-staggered100) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100) |

**Recommendation from the dated diagnosis.** Improve both precise placement and within-cluster shape, then meet the sustained checks and independent holdout. Longer training alone has not solved it.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#staggered100).


<a name="grid100_14k"></a>

## Keep the 100-cluster grid correct for longer

Test: `grid100_14k` · Required for qualification and included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**Historical diagnosis (2026-10-15).** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Historical passer evidence.** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_14k) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-grid100_14k) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_14k) |

**Recommendation from the dated diagnosis.** First obtain a genuine passing parent under the unchanged clean-sampling task, then continue that exact checkpoint. Keep rescue experiments separate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#grid100_14k).


<a name="rotated100_14k"></a>

## Keep the 100-cluster rotated grid correct for longer

Test: `rotated100_14k` · Required for qualification and included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**Historical diagnosis (2026-10-15).** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Historical passer evidence.** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-rotated100_14k) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-rotated100_14k) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-rotated100_14k) |

**Recommendation from the dated diagnosis.** First obtain a genuine passing parent under the unchanged clean-sampling task, then continue that exact checkpoint. Keep rescue experiments separate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#rotated100_14k).


<a name="staggered100_14k"></a>

## Keep the 100-cluster staggered grid correct for longer

Test: `staggered100_14k` · Required for qualification and included in full-suite results.

**What it tests.** Continue a model that passed the 7,000-update 100-cluster test to 14,000 updates without losing quality.

**Historical diagnosis (2026-10-15).** The required passing parent checkpoint does not exist. These are blocked continuations, not completed numerical failures. Continuing failed BISPF parents diagnostically still did not pass.

**Historical passer evidence.** No qualifying PASS; no passing current-declaration parent is available.

- The BISPF rescue continuations from failed parents remain below the precision gate and do not establish a valid endurance pass.
- The rotated rescue has an INVALID cross-device coverage audit near a scoring boundary; it is not a PASS or a scientific FAIL.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-staggered100_14k) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-staggered100_14k) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-staggered100_14k) |

**Recommendation from the dated diagnosis.** First obtain a genuine passing parent under the unchanged clean-sampling task, then continue that exact checkpoint. Keep rescue experiments separate.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#staggered100_14k).


<a name="target_shift_recovery"></a>

## Recover quickly after the target moves

Test: `target_shift_recovery` · Required for qualification and included in full-suite results.

**What it tests.** Learn and hold a ring of blobs, then track a sudden sideways shift within 400 updates and stay correct; a frozen model must fail to show the change is real.

**Historical diagnosis (2026-10-15).** BISPF learns and holds the original ring, then recovers too late: it becomes steady 620 updates after the shift, missing the 400-update deadline. This is an adaptation-speed failure, not an inability to recover at all.

**Historical passer evidence.** No complete PASS under the current declaration in the audited evidence.

- BISPF meets three of four parts but misses the recovery deadline.
- Older PR215/PR217 recipes pass a different ring-shift question under earlier contracts, not this test.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-target_shift_recovery) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-target_shift_recovery) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-target_shift_recovery) |

**Recommendation from the dated diagnosis.** Improve adaptation speed while keeping stationary stability. If the deadline itself is reconsidered, use an explicitly versioned task and preserve the old failure.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#target_shift_recovery).


<a name="mode_hold"></a>

## Keep every learned mode alive

Test: `mode_hold` · Required for qualification and included in full-suite results.

**What it tests.** Keep generating every target mode during continued training, rather than losing modes after initially finding them.

**Historical diagnosis (2026-10-15).** BCAP falls one sustained window short in the reviewed run (3 terminal passing checks; 5 required). This single test accounts for BISPF's six-point lead when repeated across six views.

**Historical passer evidence.** Yes: the selected BISPF recipe and some research configurations.

- `bispf-v1` has a recorded PASS with a 12-check terminal streak. Its source is stale; this is not a fresh current-source qualification.
- The cell-quota minimum-change recipe also passes mode hold but fails ring16. The reviewed tether-only BCAP ablation does not reproduce the BISPF win.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-mode_hold) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-mode_hold) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-mode_hold) |

**Recommendation from the dated diagnosis.** Preserve the BISPF pass on current source and combine it with repairs to Gaussian, vector and image failures.

Evidence: [BISPF lead and ablation evidence](WHAT_WORKED_REVIEW_20261015.md).


<a name="ring_extension"></a>

## Keep a learned ring stable for a long run

Test: `ring_extension` · Required for qualification and included in full-suite results.

**What it tests.** Acquire all eight ring blobs, hold them for 1,200 consecutive checks, then survive 300 additional checks on the same uninterrupted run.

**Historical diagnosis (2026-10-15).** BISPF lacks an ordinary qualifying run in the selected cohort. BCAP's diagnostic run loses quality during the hold phase; the extension cannot rescue a failed hold.

**Historical passer evidence.** Yes: BISPF in a saved separate-runtime diagnostic; no selected ordinary PASS.

- BISPF producer `bf5d13ff` passes the full diagnostic run: 1,200/1,200 hold checks and 300/300 extension checks, with minimum quality above 0.99.
- That diagnostic result is evidence the problem is solvable. It does not fill the selected cohort's unmeasured cell. `ring_hold` remains a visible phase, not a second independent test.

**Scope correction for retained diagnostic observations.** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring_extension) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-ring_extension) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-ring_extension) |

**Recommendation from the dated diagnosis.** Measure the unchanged BISPF recipe in its ordinary cohort on current source and preserve the whole acquisition/hold/extension protocol.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#ring_extension).


<a name="ring16_acquisition"></a>

## Learn sixteen distinct clusters from scratch

Test: `ring16_acquisition` · Required for qualification and included in full-suite results.

**What it tests.** Learn all sixteen blobs on a ring with balanced populations and real within-blob spread, rather than sixteen collapsed points.

**Historical diagnosis (2026-10-15).** The selected K3P, KA2, R1/R2 and GAN v3 measurements fail this Tier 1 gate, preventing later qualification. Their results on other questions do not erase this acquisition failure.

**Historical passer evidence.** Yes: selected BCAP and BISPF recipes.

- The selected BCAP DualNorm and `bispf-v1` records pass. Their recorded results remain bound to their original budgets and sources.
- Some research recipes pass ring16 but lose mode hold; per-test winners do not establish a single winner for the entire suite.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_acquisition) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [FAIL](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-ring16_acquisition) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_acquisition) |

**Recommendation from the dated diagnosis.** Retain the passing ring baseline while repairing the held tests. Longer-budget or changed-schedule results must remain explicitly separate until adopted and measured.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#gaussian1d_stability) · [Task declaration](../../configs/forge/tasks/ring16_acquisition.json).


<a name="clockfree_audit_measurement_v1"></a>

## Learning must not depend on external clock labels

Test: `clockfree_audit_measurement_v1` · Required for qualification and included in full-suite results.

**What it tests.** Check that the same learning state and inputs produce the same update despite changed step labels, declared horizon, evaluation cadence or a save/restart.

**Historical diagnosis (2026-10-15).** The recorded scheduled K3P, KA2, R1/R2 and GAN v3 recipes depend on step or horizon through annealing. Good numerical fits do not satisfy this separate invariance requirement.

**Historical passer evidence.** Yes: selected BCAP and BISPF recipes.

- BCAP and BISPF pass the Tier 1 measurement with constant-rate configurations.
- `clockfree_audit` is the duplicate longer-form gate and remains visible as a diagnostic; it is excluded from the distinct full-suite total to avoid counting the same question twice.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-clockfree_audit_measurement_v1) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [PASS](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-clockfree_audit_measurement_v1) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [FAIL](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-clockfree_audit_measurement_v1) |

**Recommendation from the dated diagnosis.** Use saved internal state rather than external labels for adaptation, or explicitly version the desired policy. Do not silently treat an annealed recipe as clock-free.

Evidence: [Test-by-test evidence](test-dispositions-20261015/README.md#clockfree_audit).


<a name="zero_tanh_rail_escape"></a>

## Escape a tanh saturation rail after the target shifts to zero

Test: `zero_tanh_rail_escape` · Held for qualification; included in full-suite results.

**What it tests.** First converge near the tanh saturation rail at (+0.90, -0.90) where tanh' is attenuated (~0.19), then escape the rail within 400 updates when the target shifts to (0.0, 0.0) at update 1,001 and stay settled there.

**Historical diagnosis (2026-10-15).** Bounded tanh heads attenuate pre-activation gradients near +/-1. Recipes that allow large pre-activation drift during the first 1,000 updates can lock into the rail or overshoot and oscillate around zero after the shift; no ordinary selected-cohort qualification run has been recorded on this newly added Tier 3 gate yet.

**Historical passer evidence.** Yes in standalone public-API seed-0 execution (K3P); not yet measured across the full selected-family roster.

- Standalone public-API execution of `k3p` at seed 0 passes all 5 terminal checks (`mean_abs_error = 0.00298 <= 0.005`, `rms_error = 0.00421 <= 0.01`, `p99_abs_error = 0.01416 <= 0.025`, `saturated_fraction = 0.0`).
- The 3-level solvability suite (`tests/test_solvability_tanh.py`) verifies 200/200 oracle passes and a constructive network witness.

**Scope correction for retained diagnostic observations.** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-zero_tanh_rail_escape) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [BLOCKED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-zero_tanh_rail_escape) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-zero_tanh_rail_escape) |

**Recommendation from the dated diagnosis.** Run the full Tier 3 qualified family roster (`k3p`, `ka2`, `r1r2`, `bcap-pure`, `bcap-dualnorm`, `atlas`, `e22`, `release07-gan-v3`) in ordinary cohorts and promote from held to required once a global recipe passes it together with all counted Tier 1-3 requirements.

Evidence: [Public-API readout](../toy_audit/api_contract/api-zero-tanh-rail-escape/readout.md) · [Task declaration](../../configs/forge/tasks/zero_tanh_rail_escape.json).


<a name="grid100_mlp16"></a>

## Survive a 16-layer unnormalized MLP generator on the 100-cluster grid

Test: `grid100_mlp16` · Held for qualification; included in full-suite results.

**What it tests.** Learn all 100 equal-weight Gaussian modes on `grid100` using a 16-hidden-layer 128-wide LeakyReLU(0.2) MLP generator without residual skips or normalization layers.

**Historical diagnosis (2026-10-15).** Increasing generator depth from 3 to 16 hidden layers contracts initial output spread by ~156x and attenuates first-layer and particle-prior gradients by ~73x at step 1. In the standalone seed-0 `k3p` run, the 16-layer generator collapses to 0/100 modes (`output_sigma = 0.029`, `mass_tv = 0.99`).

**Historical passer evidence.** No full PASS under the current clean-sampling declaration; constructive capacity witness passes.

- Constructive 16-layer capacity witness in `tests/test_solvability_deep100g.py` recovers 100/100 modes and passes all coverage and 100k accuracy bounds, proving the 16-layer architecture can represent the target.
- Oracle sampler passes 200/200 draws with >= 2x noise-floor headroom.

**Scope correction for retained diagnostic observations.** Import-time shared-generator replacement and residual-generator alias do not establish the originally declared architecture. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_mlp16) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [BLOCKED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-grid100_mlp16) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_mlp16) |

**Recommendation from the dated diagnosis.** Pair layer-wise gradient normalization (such as `bcap-nsgda-layer` or `bcap-dualnorm`) with balanced prior-to-generator step scaling so early updates do not collapse the 16-layer Jacobian before particles separate across the 10x10 grid.

Evidence: [Public-API readout](../toy_audit/api_contract/api-grid100-mlp16/readout.md) · [Task declaration](../../configs/forge/tasks/grid100_mlp16.json).


<a name="grid100_res16_tanh"></a>

## Learn the 100-cluster grid with a 16-layer residual MLP and bounded tanh head

Test: `grid100_res16_tanh` · Held for qualification; included in full-suite results.

**What it tests.** Recover all 100 Gaussian modes on `grid100` using a 16-layer identity-skip residual MLP generator and a bounded `S=5.25` scaled `tanh` output head without saturating the `|x| > 0.99 * 5.25` rails.

**Historical diagnosis (2026-10-15).** Identity skip connections prevent the 16-layer vanishing-gradient collapse of `grid100_mlp16`, but under unscaled critic updates (`k3p` at seed 0) 96.4% of generated samples are pushed into the `tanh` saturation rails (`saturated_fraction = 0.9639` vs. `<= 0.001` bound, `modes = 0/100`).

**Historical passer evidence.** No full PASS on the unnormalized backbone; paired `grid100_res16_scaled_tanh` task (`ScaledResidualMLPGenerator` + `TanhRailGuard`, 8/8 selected families PASS at full 7,000-step GPU budget) and constructive residual-tanh witness pass with zero saturation.

- Paired `grid100_res16_scaled_tanh` (`configs/forge/tasks/grid100_res16_scaled_tanh.json`) with `ScaledResidualMLPGenerator(use_norm=True, preserve_latent_norm=True)` and `TanhRailGuard` passes all 8 selected families (`atlas`, `bcap-pure`, `bispf`, `e22`, `k3p`, `ka2`, `r1r2`, `release07-gan-v3`) at full 7,000-step GPU budget (`100/100` modes, `saturated_fraction = 0.0 <= 0.001`), isolating trainer depth transport from unscaled initialization pathology.
- Constructive 16-layer residual + `S=5.25` tanh witness in `tests/test_solvability_deep100g.py` passes all 100-mode coverage, 100k accuracy, and `saturated_fraction <= 0.001` bounds (`saturated_fraction = 0.0`).
- Oracle sampler passes 200/200 draws.

**Scope correction for retained diagnostic observations.** Import-time shared-generator replacement and residual-generator alias do not establish the originally declared architecture. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-grid100_res16_tanh) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [BLOCKED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-grid100_res16_tanh) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-grid100_res16_tanh) |

**Recommendation from the dated diagnosis.** Keep `grid100_mlp16` and `grid100_res16_tanh` as unnormalized trainer stress tests alongside the paired `grid100_res16_scaled_tanh` control, and control pre-tanh logit growth on unnormalized backbones via bounded step norms.

Evidence: [Public-API readout](../toy_audit/api_contract/api-grid100-res16-tanh/readout.md) · [Task declaration](../../configs/forge/tasks/grid100_res16_tanh.json) · [Paired scaled-backbone declaration](../../configs/forge/tasks/grid100_res16_scaled_tanh.json).


<a name="misgan_block_recovery"></a>

## Recover an 8D lifted 100-mode manifold from incomplete block-masked data (MisGAN + ppost)

Test: `misgan_block_recovery` · Held for qualification; included in full-suite results.

**What it tests.** Train a 2-pair incomplete-data MisGAN (`(G_x, D_x)` + `(G_m, D_m)` with detached mask gradient in `L_x`) observing only masked 8D data `(x * m, m)` under structured block dropout, then impute held-out rows via native Bayesian particle-posterior (`ppost`) resampling from `G_x`.

**Historical diagnosis (2026-10-15).** When 10% of rows observe only sensor `{7}`, unanchored 8D critic updates without partial-mask hints or observed-subspace completion suffer from mode-averaging and off-plane drift (`off_plane = 3.55` vs. `<= 0.04`, `modes = 3/100`).

**Historical passer evidence.** Yes: all 8 selected families (`atlas`, `bcap-pure`, `bispf`, `e22`, `k3p`, `ka2`, `r1r2`, `release07-gan-v3`) pass in the formulaic coverage matrix via GAIN-style hint conditioning on `D_x` and observed-subspace completion without temperature annealing.

- All 8 selected families (`atlas`, `bcap-pure`, `bispf`, `e22`, `k3p`, `ka2`, `r1r2`, `release07-gan-v3`) pass all 5 sustained terminal checks on both `misgan_block_recovery` and `misgan_mcar_p50_recovery` with decoupled `prior_x`/`prior_m`, GAIN-style hint conditioning on `D_x` (`hint_dim=8`), static `TemperatureSigmoid(0.66)`, and observed-subspace completion.
- Constructive witness and `origin/misgan-toy` detached-mask `G_x` + `ppost` achieve 100/100 modes, `hq >= 0.99`, and `acc_lo = 0.759 >= 0.70` (with 2.88x noise-floor headroom relative to the 0.777 Bayes ceiling).
- Oracle `bayes_posterior` + `ppost` passes 200/200 draws in `tests/test_solvability_misgan.py`.

**Scope correction for retained diagnostic observations.** Observed-subspace solver overwrites generators/prior tables; neural-imputer observation may use posterior resampling instead of G_i. EMA sampling and complete-state restoration also differ from the declared live contract. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_recovery) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [BLOCKED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-misgan_block_recovery) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_recovery) |

**Recommendation from the dated diagnosis.** Promote `misgan_block_recovery` and `misgan_mcar_p50_recovery` from held to required once an updated global qualification cohort is registered.

Evidence: [Public-API readout](../toy_audit/api_contract/api-misgan-block-recovery/readout.md) · [Task declaration](../../configs/forge/tasks/misgan_block_recovery.json).


<a name="misgan_block_neural_imputer"></a>

## Learn a multimodal neural imputer on ambiguous single-sensor block masks without ppost

Test: `misgan_block_neural_imputer` · Held for qualification; included in full-suite results.

**What it tests.** Train a 3-pair MisGAN (`(G_x, D_x)` + `(G_m, D_m)` + `(G_i, D_i)`) and grade the neural imputer `G_i([x * m, m, z_i])` directly on the 16 held-out ambiguous single-sensor queries without `ppost` particle resampling.

**Historical diagnosis (2026-10-15).** Standard neural imputers `G_i([x * m, m, z_i])` satisfy the observed-sensor identity (`observed_max_error = 0.0`) and global missing-coordinate spread (`missing_std_ratio = 0.803` in `[0.7, 1.3]`), but suffer from conditionalGAN `z_i` noise-channel ignore on the 10% single-sensor rows without observed-subspace coupling and `D_x` hint conditioning.

**Historical passer evidence.** Yes: all 8 selected families (`atlas`, `bcap-pure`, `bispf`, `e22`, `k3p`, `ka2`, `r1r2`, `release07-gan-v3`) pass in the formulaic coverage matrix without temperature annealing.

- All 8 selected families (`atlas`, `bcap-pure`, `bispf`, `e22`, `k3p`, `ka2`, `r1r2`, `release07-gan-v3`) pass all 5 sustained terminal checks on `misgan_block_neural_imputer` (`observed_max_error = 0.0`, `acc >= 0.95`, `acc_lo >= 0.65`, `missing_std_ratio` in `[0.70, 1.30]`) with decoupled `prior_x`/`prior_m`/`prior_i`, GAIN-style hint conditioning on `D_x`, and static `TemperatureSigmoid(0.66)`.
- Trained PyTorch neural network witness `G_i` in `tests/test_solvability_misgan.py` passes 100% of gates, proving the neural imputer architecture has sufficient capacity.
- Oracle posterior sampler passes 200/200 draws with >= 2x noise headroom.

**Scope correction for retained diagnostic observations.** Observed-subspace solver overwrites generators/prior tables; neural-imputer observation may use posterior resampling instead of G_i. EMA sampling and complete-state restoration also differ from the declared live contract. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-misgan_block_neural_imputer) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [BLOCKED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-misgan_block_neural_imputer) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-misgan_block_neural_imputer) |

**Recommendation from the dated diagnosis.** Promote `misgan_block_neural_imputer` from held to required once an updated global qualification cohort is registered.

Evidence: [Public-API readout](../toy_audit/api_contract/api-misgan-block-neural-imputer/readout.md) · [Task declaration](../../configs/forge/tasks/misgan_block_neural_imputer.json).


<a name="ring16_bigan_continuous"></a>

## Learn a continuous 16-mode ring joint Encoder-Generator-Critic BiGAN

Test: `ring16_bigan_continuous` · Held for qualification; included in full-suite results.

**What it tests.** Train a continuous 2D multimodal joint BiGAN `(G(z), E(x), D(x, z))` on the 16-Gaussian ring (`MoGParticlePrior(sigma=0.1)`), simultaneously satisfying forward 16-mode generation and paired round-trip reconstruction `x -> G(E(x))`.

**Historical diagnosis (2026-10-15).** In the standalone seed-0 `bcap` public-API run without cycle-consistency or kernel-augmented critics, the joint game discovers all 16/16 modes (`modes = 16`) and achieves high round-trip mode consistency (`mode_consistency = 0.9951 >= 0.98`), but within-mode variance remains too broad (`hq = 0.238` vs. `>= 0.85`, `recon_rmse = 0.220` vs. `<= 0.12`).

**Historical passer evidence.** Yes: `k3p` (adopted as the default continuous BiGAN recipe) and `ka2` paired with ALICE cycle-consistency (`lambda_cycle = 2.0`) and coordinate Fourier features (`K=3`) pass all 5 sustained terminal checks.

- `k3p` (default recipe on `api-ring16-bigan-continuous` and `api-alice-spiral-bigan`) and `ka2` pass all 5 sustained terminal checks (`modes = 16`, `hq >= 0.85`, `mass_tv <= 0.15`, `recon_rmse <= 0.12`, `mode_consistency >= 0.98`) in the formulaic coverage matrix.
- Trained PyTorch `(G, E)` witness in `tests/test_solvability_bigan.py` passes all forward generation and round-trip reconstruction gates.
- Oracle joint sampler passes 200/200 draws with >= 2x noise headroom.

**Scope correction for retained diagnostic observations.** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [BLOCKED](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-ring16_bigan_continuous) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [BLOCKED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-ring16_bigan_continuous) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-ring16_bigan_continuous) |

**Recommendation from the dated diagnosis.** Extend kernel-augmented joint `(x, z)` curvature regularization and cycle-consistency conditioning to the remaining non-kernel trainer families so all selected families contract within-mode variance without losing 16-mode coverage.

Evidence: [Public-API readout](../toy_audit/api_contract/api-ring16-bigan-continuous/readout.md) · [Task declaration](../../configs/forge/tasks/ring16_bigan_continuous.json).


<a name="two_pole"></a>

## Separate direct sample particles across two symmetric poles

Test: `two_pole` · Required for qualification and included in full-suite results.

**What it tests.** Drive a generator-free direct particle cloud from zero to two symmetric poles (`mean_abs >= 0.30`) and hold them steady with bounded critic gradient (`grad_med <= 1.0`) across the final five checks.

**Historical diagnosis (2026-10-15).** In selected `atlas-particle-normal-v4` (`particles` + `NormalNoiseManipulator(sigma=0.0065)` + `birth_death_cells=256`), direct particle coordinates without a separate generator remain pinned near the origin (`mean_abs = 0.00242` vs. `>= 0.30`, `grad_med = 0.00371`). Critic-space kNN birth-death (`birth_death_cells = 64` capped on the direct particle batch) recycles particles around the origin before symmetric pole separation develops.

**Historical passer evidence.** Yes: selected BCAP, BISPF, K3P, KA2, R1/R2 and Release-07 GAN v3 recipes pass.

- Selected `recipe-prior-refactor-bcap-pure-v1`, `bispf-v1`, `recipe-prior-refactor-k3p-v1`, `recipe-prior-refactor-ka2-v1`, `recipe-prior-refactor-r1r2-v1` and `recipe-prior-refactor-release07-gan-v3-v1` pass the ordinary Tier 1 gate.
- E22 remains blocked on direct particle-cloud host compatibility.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-two_pole) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-two_pole) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [FAIL](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-two_pole) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [PASS](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [PASS](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [PASS](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [PASS](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-two_pole) |

**Recommendation from the dated diagnosis.** Gate critic-space particle birth-death when no separate generator module is present (direct particle-transport hosts) or prevent real-anchor cloning from collapsing bipolar symmetry near zero, while preserving `atlas-particle-normal-v4`'s 24-pass full-suite gains.

Evidence: [Test dispositions and policy-cloud notes](test-dispositions-20261015/README.md) · [Task declaration](../../configs/forge/tasks/two_pole.json).


<a name="five_word_joint_smoke"></a>

## Acquire five discrete words and paired BiGAN reconstructions in 20,001 joint updates

Test: `five_word_joint_smoke` · Required for qualification and included in full-suite results.

**What it tests.** Train a joint generator-encoder-prior-critic BiGAN over five five-letter words for 20,001 updates, acquiring all five clean word modes (`modes == 5`, `quality_fraction >= 0.95`, `mass_tv <= 0.10`) and exact high-confidence paired reconstructions (`reconstruction_exact == 1`, `minimum_reconstruction_token_probability >= 0.90`) at an independently confirmed scheduled state.

**Historical diagnosis (2026-10-15).** While selected `atlas-particle-normal-v4` passes `ae_gan_hold` and `unused_token_hold`, on the 20,001-step five-word joint BiGAN game its `particles` + `NormalNoiseManipulator(sigma=0.0065)` + `birth_death_cells=256` recipe fails to acquire discrete word modes (`modes = 0/5`, `quality_fraction = 0.0`, `reconstruction_exact = 0`, `reconstruction_nll = 22.46`).

**Historical passer evidence.** Yes: selected BCAP, BISPF, K3P, KA2, R1/R2 and Release-07 GAN v3 recipes pass.

- Selected `bispf-v1`, `recipe-prior-refactor-bcap-pure-v1`, `recipe-prior-refactor-k3p-v1`, `recipe-prior-refactor-ka2-v1`, `recipe-prior-refactor-r1r2-v1` and `recipe-prior-refactor-release07-gan-v3-v1` pass the Tier 1 smoke gate.
- Passing `five_word_joint_smoke` certifies one confirmed acquisition state; `five_word_joint_hold` separately grades sustained post-acquisition stability.

**Scope correction for retained diagnostic observations.** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_smoke) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [FAIL](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-five_word_joint_smoke) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [PASS](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [PASS](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [PASS](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [PASS](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_smoke) |

**Recommendation from the dated diagnosis.** Diagnose how critic-space birth-death and `sigma=0.0065` latent perturbation interact with the joint `(x, z)` BiGAN encoder-generator coupling over 20k discrete-token updates without regressing continuous multimodal coverage.

Evidence: [Five-word tier split readout](five-word-tier-split/README.md) · [Task declaration](../../configs/forge/tasks/five_word_joint_smoke.json).


<a name="model_ae_gan_multimodal"></a>

## Reconstruct four noisy corner Gaussians and generate balanced independent modes (AE-GAN)

Test: `model_ae_gan_multimodal` · Required for qualification and included in full-suite results.

**What it tests.** Train a public-API multimodal AE-GAN to reconstruct four equally weighted corner Gaussians (`reconstruction_mse <= 0.025`, `reconstruction_mode_agreement >= 0.98`) while independently generating all four modes with balanced masses (`mass_tv <= 0.15`) and well-conditioned within-mode covariances (`0.10 <= eigen_ratio <= 4.0`) across the final five checks.

**Historical diagnosis (2026-10-15).** Selected `atlas-particle-normal-v4` satisfies seven of eight terminal bounds (`modes = 4`, `precision = 0.9980 >= 0.90`, `reconstruction_mse = 0.00438 <= 0.025`, `reconstruction_mode_agreement = 1.0`, `component_min_eigen_ratio = 0.140 >= 0.10`, `component_max_eigen_ratio = 1.169 <= 4.0`, `component_center_rms_sigma = 0.571 <= 1.0`), but misses the independent mode-mass balance gate (`mass_tv = 0.1948` vs. `<= 0.15`). Other selected families remain unmeasured in ordinary CUDA cohorts on this Rev 10+ Tier 1 task.

**Historical passer evidence.** Yes in standalone bounded API research (`bispf-optimized-quintic-v1` CPU cohort); no ordinary selected CUDA PASS yet.

- Standalone bounded API research (`bispf-optimized-quintic-v1` on CPU) passes all eight sustained gates (`reconstruction_mse = 0.00927`, `precision = 0.9968`, `mass_tv = 0.1177 <= 0.15`).
- That CPU research cohort is separate from the ordinary CUDA selected-family rows and does not fill `bispf-v1`'s unmeasured cell.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [PASS](families/empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_gan_multimodal) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [FAIL](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-model_ae_gan_multimodal) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_gan_multimodal) |

**Recommendation from the dated diagnosis.** Reduce prior mass imbalance under joint AE-GAN encoding (for example, balancing birth-death cell quotas against encoder-posterior assignments) and record ordinary CUDA measurements for the remaining selected families.

Evidence: [AE-GAN public-API model report](api-model-toys/ae.md) · [Task declaration](../../configs/forge/tasks/model_ae_gan_multimodal.json).


<a name="model_ddgan_gaussian_chain"></a>

## Learn four-step conditional reverse transitions and ancestral generation (DDGAN)

Test: `model_ddgan_gaussian_chain` · Required for qualification and included in full-suite results.

**What it tests.** Train a four-step public-API DDGAN on a 2D Gaussian forward diffusion chain so every reverse conditional transition (`t1..t4`, `q0`/`q1`) and the final ancestral marginal satisfy `projection_cdf <= 0.12`, `normalized_mean <= 0.25`, and covariance eigenvalues in `[0.65, 1.40]` across the last five checks.

**Historical diagnosis (2026-10-15).** Both selected `atlas-particle-normal-v4` (ordinary CUDA) and the bounded `bispf-optimized-quintic-v1` API research run (CPU) match the outer reverse steps (`t3`, `t4`), but miss the sharp final denoising transitions (`t1`, `t2`) and ancestral marginal (`atlas-particle-normal-v4`: `final_projection_cdf = 0.1695 > 0.12`, `final_normalized_mean = 0.3367 > 0.25`, `t1_q0_projection_cdf = 0.2747 > 0.12`, `t1_q0_covariance_eigen_min = 0.2710 < 0.65`).

**Historical passer evidence.** No PASS under the current declaration in either ordinary selected cohorts or bounded API research.

- Oracle transition samplers pass all 38 threshold checks in the bounded protocol, confirming scorer calibration.
- Saved-state diagnostics isolate the failure to conditional reverse-step variance and mean drift at `t=1, 2` rather than numerical instability.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ddgan_gaussian_chain) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [FAIL](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-model_ddgan_gaussian_chain) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-model_ddgan_gaussian_chain) |

**Recommendation from the dated diagnosis.** Improve per-timestep conditional critic discrimination and reverse-generator stochasticity at small `t` under one unchanged global recipe without per-task hyperparameter tuning.

Evidence: [DDGAN public-API model report](api-model-toys/ddgan.md) · [DDGAN saved-state failure investigation](ddgan-failure-investigation/README.md) · [Task declaration](../../configs/forge/tasks/model_ddgan_gaussian_chain.json).


<a name="model_ae_ddgan_gaussian_chain"></a>

## Pair particle AE reconstruction with four-step DDGAN reverse transitions and ancestral generation

Test: `model_ae_ddgan_gaussian_chain` · Required for qualification and included in full-suite results.

**What it tests.** Reconstruct held-out inputs through a learned particle autoencoder and four public DDGAN reverse steps (`paired_nmse <= 0.10`, `paired_worst_axis_nrmse <= 0.25`, `paired_shuffled_gain >= 3.0`) while simultaneously passing all conditional reverse-transition and independent ancestral-chain bounds.

**Historical diagnosis (2026-10-15).** Selected `atlas-particle-normal-v4` passes every paired four-step reconstruction gate (`paired_nmse = 0.0147 <= 0.10`, `paired_worst_axis_nrmse = 0.1333 <= 0.25`, `paired_shuffled_gain = 134.67 >= 3.0`) and `final_projection_cdf = 0.1199 <= 0.12`, matching the bounded `bispf-optimized-quintic-v1` research pattern where paired encoding succeeds (`paired_nmse = 0.0441`) while independent ancestral covariance (`final_covariance_eigen_min = 0.4226 < 0.65`) and `t1`/`t2` reverse-transition bounds fail (`t1_q0_projection_cdf = 0.5950 > 0.12`).

**Historical passer evidence.** No complete PASS under the current declaration; paired four-step reconstruction passes while independent/conditional reverse generation fails.

- Both `atlas-particle-normal-v4` (ordinary CUDA) and `bispf-optimized-quintic-v1` (bounded CPU research) pass all four paired four-step reconstruction gates with >45x-134x shuffled-input error gain.
- Oracle reverse-chain controls pass all 42 gates.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-model_ae_ddgan_gaussian_chain) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [FAIL](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-model_ae_ddgan_gaussian_chain) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-model_ae_ddgan_gaussian_chain) |

**Recommendation from the dated diagnosis.** Align the ancestral prior and reverse chain with the encoder latent manifold at `t=1, 2` so independent ancestral sampling retains full variance (`>= 0.65`) without degrading paired reconstruction.

Evidence: [AE-DDGAN public-API model report](api-model-toys/ae-ddgan.md) · [Task declaration](../../configs/forge/tasks/model_ae_ddgan_gaussian_chain.json).


<a name="precision_gaussian_mxfp8"></a>

## Learn a Gaussian with 8-bit MXFP8 training arithmetic

Test: `precision_gaussian_mxfp8` · Held for qualification; included in full-suite results.

**What it tests.** Train from the shared deterministic initialization using Blackwell-format MXFP8 software-reference forward and backward GEMM operands; match the full scalar Gaussian law at five final checks.

**Historical diagnosis (2026-10-15).** No full-budget family evidence has been registered. A matched FP32 diagnostic separates baseline learning failures from precision sensitivity. Native kernel and higher-order autograd parity remain unverified.

**Historical passer evidence.** Not measured. Short software checks do not establish a quality PASS.

- No compatible full-budget passing configuration is recorded.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_mxfp8) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-precision_gaussian_mxfp8) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_mxfp8) |

**Recommendation from the dated diagnosis.** Run the unchanged selected family recipes through the optional Tier 1 tasks and their FP32 control under the expanded reservation budget. Diagnose failed bounds before changing one global recipe; native validation is a separate step.

Evidence: [Protocol, precision semantics and limitations](precision-stretch/README.md#precision-stretch).


<a name="precision_gaussian_nvfp4"></a>

## Learn a Gaussian with 4-bit NVFP4 training arithmetic

Test: `precision_gaussian_nvfp4` · Held for qualification; included in full-suite results.

**What it tests.** Train from the shared deterministic initialization using Blackwell-format NVFP4 software-reference forward and backward GEMM operands; match the full scalar Gaussian law at five final checks.

**Historical diagnosis (2026-10-15).** No full-budget family evidence has been registered. A matched FP32 diagnostic separates baseline learning failures from precision sensitivity. Native kernel and higher-order autograd parity remain unverified.

**Historical passer evidence.** Not measured. Short software checks do not establish a quality PASS.

- No compatible full-budget passing configuration is recorded.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-precision_gaussian_nvfp4) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-precision_gaussian_nvfp4) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-precision_gaussian_nvfp4) |

**Recommendation from the dated diagnosis.** Run the unchanged selected family recipes through the optional Tier 1 tasks and their FP32 control under the expanded reservation budget. Diagnose failed bounds before changing one global recipe; native validation is a separate step.

Evidence: [Protocol, precision semantics and limitations](precision-stretch/README.md#precision-stretch).


<a name="five_word_joint_hold"></a>

## Hold all five discrete words and exact BiGAN inversion across 4,000 continuation updates

Test: `five_word_joint_hold` · Required for qualification and included in full-suite results.

**What it tests.** Restore the candidate's own confirmed five-word acquisition checkpoint in a fresh process and pass every generation and exact encoder-inversion bound across all 24 scheduled checks over 4,000 additional updates.

**Historical diagnosis (2026-10-15).** Selected `empf-v7` finishes the 4,000-step continuation inside every terminal bound (`modes = 5`, `quality_fraction = 1.0`, `mass_tv = 0.0346 <= 0.10`, `reconstruction_exact = 1`, `minimum_reconstruction_token_probability = 0.9993 >= 0.90`), but dips on an intermediate scheduled check during post-acquisition settling under the size-gated `newton_schulz_hybrid_v1` dense DualNorm profile.

**Historical passer evidence.** Yes: selected BISPF (`bispf-optimized-quintic-v1`) and historical `empf-v2` pass all 24 continuation checks.

- Selected `bispf-optimized-quintic-v1` passes all 24 scheduled continuation checks from its own smoke checkpoint, while failing `gaussian1d_stability`.
- Historical `empf-v2` (`readout-05eaed46ab0e45214ae8e7b4.json`) also passes all 24 continuation checks under exact `polar_factor` SVD, trading `gaussian1d_stability` and `precision_gaussian_fp32`.

**Scope correction for retained diagnostic observations.** Formulaic runner changes selected recipe noise settings or E22 mechanisms without a distinct adapted-candidate identity and complete resolved-recipe/sampling witness. Original outcomes remain archived; they do not establish the declared unassisted recipe/task claim. [Exact observation identities](evidence-corrections-2026-11-08.json).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [PASS](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [FAIL](families/empf.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-five_word_joint_hold) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-five_word_joint_hold) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [BLOCKED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-five_word_joint_hold) |

**Recommendation from the dated diagnosis.** Damp transient post-acquisition joint-critic excursions on the discrete 5-particle codebook without losing `empf-v7`'s `gaussian1d_stability`, `precision_gaussian_fp32`, and `five_word_joint_smoke` wall-time budget.

Evidence: [EMPF-v7 ordinary readout receipt](records/readout-4901963885d977da23e139ba.json) · [Task declaration](../../configs/forge/tasks/five_word_joint_hold.json).


<a name="interp_sparse_arc"></a>

## Interpolate unseen latent k-NN chords along a 270-degree circular arc from 12 zero-sigma particles without MoG

Test: `interp_sparse_arc` · Held for qualification; included in full-suite results.

**What it tests.** Train 12 discrete zero-sigma `ParticlePrior` locations and an MLP generator on a 2D circular arc (`R = 2.0`, `sigma = 0.03`) under protocol seed 0 so both endpoints (`endpoint_hq >= 0.85`) and unseen latent k-NN chords (`interp_hq >= 0.80`, `interp_arc_bins >= 22`, `midpoint_novelty >= 0.20`) stay on the curved manifold across five final live checks.

**Historical diagnosis (2026-10-15).** No selected recipe sustains all five final live checks (`0/8` runnable, `e22` preflight `BLOCKED` by `live_sampling`). Selected `atlas-particle-normal-v4` meets every bound at step 600 (`endpoint_hq = 1.000`, `interp_hq = 0.9365`, `output_knn_interp_hq = 1.000`, `30/32` bins, `midpoint_novelty = 0.576`) and passes `3/5` terminal checks via its recipe-owned `normal_noise` (`0.0065`) prior manipulator, while `empf-v7` bends interior chords (`interp_hq = 0.8254`) with off-arc endpoints (`endpoint_hq = 0.6667 < 0.85`) and classical Adam families (`release07-gan-v3`, `r1r2`, `k3p`, `ka2`) place endpoints on the arc (`0.83-1.00`) but cut straight Euclidean secants across unseen chords (`interp_hq = 0.52-0.65`).

**Historical passer evidence.** No sustained 5-check PASS in the diagnostic stretch campaign; `atlas-particle-normal-v4` passes 3/5 final checks (`interp_hq = 0.9365` at step 600).

- No selected family achieves a 5/5 sustained live PASS under the current declaration.
- Diagnostic stretch run `atlas-particle-normal-v4` (`readout-669ead9699347bdcc998eb81.json`) passes 3/5 terminal checks and all final-step bounds; analytic smooth-arc oracle passes all gates while chord-cutter and Voronoi-snap controls fail as designed.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [FAIL](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_sparse_arc) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-interp_sparse_arc) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-interp_sparse_arc) |

**Recommendation from the dated diagnosis.** Add a recipe-owned latent chord perturbation or local Jacobian smoothness regularizer and damp live-iterate oscillation on zero-sigma `ParticlePrior` without sacrificing counted Tier 1-3 passes.

Evidence: [Particle interpolation stretch report and 9-family table](particle-interpolation-stretch/README.md#interp_sparse_arc) · [Atlas diagnostic stretch readout receipt](records/readout-669ead9699347bdcc998eb81.json) · [Task declaration](../../configs/forge/tasks/interp_sparse_arc.json).


<a name="interp_img_patch_shift"></a>

## Translate a sharp 2x2 patch smoothly across unseen latent chords in 8x8 images from 16 zero-sigma particles

Test: `interp_img_patch_shift` · Held for qualification; included in full-suite results.

**What it tests.** Train 16 zero-sigma `ParticlePrior` codes (`z_dim = 4`) and a `residual_upsample` convolutional generator on continuous single-patch 8x8 images so unseen latent k-NN chords render single sharp shifted patches (`interp_rmse <= 0.10`, `interp_peak >= 0.78`, `midpoint_novelty >= 0.18`, `quadrant_coverage == 4`) rather than 0.5-peak pixel double exposures.

**Historical diagnosis (2026-10-15).** Archived ordinary family cohorts predate this task (`UNKNOWN`), and `e22` is preflight `BLOCKED` on clean/live scoring (`live_sampling`). In the matched CUDA stretch campaign (`particle-interp-stretch-v1`), `empf-v7` collapses to zero output (`interp_peak = 0.0`), `bispf-optimized-quintic-v1` dims chord midpoints (`interp_peak = 0.6189 < 0.78`), and `recipe-prior-refactor-bcap-pure-v1` underfits (`interp_peak = 0.3015`), whereas all five remaining families pass 5/5 terminal checks.

**Historical passer evidence.** 0 of 9 in archived ordinary cohorts (`UNKNOWN`); 5 of 8 runnable selected recipes (`atlas`, `release07-gan-v3`, `ka2`, `k3p`, `r1r2`) pass 5/5 terminal checks in matched CUDA stretch diagnostics.

- Separate CUDA stretch diagnostics (`particle-interp-stretch-v1`) pass 5/5 terminal checks for `atlas-particle-normal-v4` (`interp_rmse = 0.0250`, `interp_peak = 0.9358`), `recipe-prior-refactor-ka2-v1` (`0.0367`, `0.9149`), `recipe-prior-refactor-k3p-v1` (`0.0381`, `0.9349`), `recipe-prior-refactor-release07-gan-v3-v1` (`0.0477`, `0.8727`), and `recipe-prior-refactor-r1r2-v1` (`0.0481`, `0.8876`).

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_img_patch_shift) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-interp_img_patch_shift) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-interp_img_patch_shift) |

**Recommendation from the dated diagnosis.** Diagnose DualNorm (`empf-v7`) zero-output collapse and `bispf` midpoint peak attenuation on 16-particle zero-sigma convolutional image generation while retaining counted discrete-word and ring passes.

Evidence: [Particle interpolation stretch report and 9-family table](particle-interpolation-stretch/README.md#interp_img_patch_shift) · [Atlas diagnostic stretch readout receipt](records/readout-669ead9699347bdcc998eb81.json) · [EMPF diagnostic stretch readout receipt](records/readout-a211bf674ae6989087e57402.json) · [Task declaration](../../configs/forge/tasks/interp_img_patch_shift.json).


<a name="interp_saddle_barycentric"></a>

## Interpolate unseen 2D latent triangle barycenters across a curved 3D saddle sheet from 24 zero-sigma particles

Test: `interp_saddle_barycentric` · Held for qualification; included in full-suite results.

**What it tests.** Train 24 zero-sigma `ParticlePrior` locations (`z_dim = 2`) and a 3D-output MLP generator on $w = 0.8(x^2 - y^2) + 0.9\cos(0.75\pi x)\cos(0.75\pi y)$ so unseen 2D latent k-NN triangle barycenters stay on the curved surface (`barycenter_surface_mae <= 0.09`, `barycenter_hq >= 0.78`, `xy_bins_covered >= 12`, `barycenter_novelty >= 0.18`) across five final live checks.

**Historical diagnosis (2026-10-15).** No selected recipe sustains all five final live checks (`0/8` runnable, `e22` preflight `BLOCKED`). `atlas-particle-normal-v4` passes `3/5` final checks and all step-800 bounds (`barycenter_surface_mae = 0.0552`, `barycenter_hq = 0.8701`, `16/16` xy bins), `release07-gan-v3` passes the final `2/5` checks (`barycenter_surface_mae = 0.0587`, `barycenter_hq = 0.8056`), and `ka2` (`barycenter_hq = 0.7639`), `bispf` (`0.7286`), and `k3p` (`0.6974`) satisfy `barycenter_surface_mae <= 0.09` while falling just short on `barycenter_hq >= 0.78`.

**Historical passer evidence.** No sustained 5-check PASS in the diagnostic stretch campaign; `atlas` (3/5 terminal checks) and `release07-gan-v3` (2/5 terminal checks) reach full single-check validity by step 800.

- No selected family achieves a 5/5 sustained live PASS under the current declaration.
- Diagnostic stretch runs `atlas-particle-normal-v4` (`readout-669ead9699347bdcc998eb81.json`, `barycenter_hq = 0.8701`) and `recipe-prior-refactor-release07-gan-v3-v1` (`readout-e838608cfd803f50b58f4e21.json`, `barycenter_hq = 0.8056`) pass every bound at step 800.

**Selected recipes' recorded results:**

| Selected recipe | Result |
| --- | --- |
| [BCAP](families/bcap-pure.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |
| [BISPF](families/bispf.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/bispf.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |
| [EMPF](families/empf.md#cohort-cuda-23a94f16ca88) | [NOT MEASURED](families/empf.md#cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric) |
| [RHBI](families/rhbi.md#cohort-cuda-23a94f16ca88) | [PASS](families/rhbi.md#cohort-cuda-23a94f16ca88-experiment-interp_saddle_barycentric) |
| [Atlas](families/atlas.md#cohort-cuda-bc3debd3fa5e) | [NOT MEASURED](families/atlas.md#cohort-cuda-bc3debd3fa5e-experiment-interp_saddle_barycentric) |
| [E22](families/e22.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/e22.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |
| [GAN v3 release 0.7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |
| [K3P](families/k3p.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/k3p.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |
| [KA2](families/ka2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/ka2.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |
| [R1/R2](families/r1r2.md#cohort-cuda-1bf9d7d34422) | [NOT MEASURED](families/r1r2.md#cohort-cuda-1bf9d7d34422-experiment-interp_saddle_barycentric) |

**Recommendation from the dated diagnosis.** Stabilize late-step live-iterate fluctuations and test recipe-owned 2D simplex interior supervision so `atlas` and `release07-gan-v3` hold `barycenter_hq >= 0.78` across all five terminal checks.

Evidence: [Particle interpolation stretch report and 9-family table](particle-interpolation-stretch/README.md#interp_saddle_barycentric) · [Atlas diagnostic stretch readout receipt](records/readout-669ead9699347bdcc998eb81.json) · [GAN v3 release 0.7 diagnostic stretch readout receipt](records/readout-e838608cfd803f50b58f4e21.json) · [Task declaration](../../configs/forge/tasks/interp_saddle_barycentric.json).
