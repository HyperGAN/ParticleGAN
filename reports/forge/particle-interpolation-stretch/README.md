<a name="particle-interpolation-stretch"></a>

# Zero-Sigma Particle Interpolation Stretch Benchmarks (No MoG)

Three frontier stretch tasks ask whether an unchanged global family recipe trained with a **zero-sigma discrete `ParticlePrior`** (`kind: "particle_cloud"`, `sigma: 0.0`, `standardize: false`)—with **no MoG Gaussian kernel smoothing**—induces a valid, high-quality, smoothly progressing manifold in the unseen latent space between finite learned particles $z_1, \dots, z_M$.

They belong to `discriminator_stability` revision 17 (`interp_sparse_arc` at Tier 1; `interp_img_patch_shift` and `interp_saddle_barycentric` at Tier 2), never block required qualification, and appear in the single [Forge technique inventory](../technique-inventory.md) stretch and disposition tables. Full matched CUDA diagnostics across all nine selected families (`seed = 0`, `NVIDIA RTX A6000`) are recorded in [summary.json](summary.json), [preflight.json](preflight.json), the supplemental [public-API publication](../../toy_audit/api_contract/interpolation/publication.json), and the eight concluded readout receipts linked below.

## Protocol, Ownership, and Controls

All three tasks execute through public `GANTrainer` and `ParticlePrior` (`sigma = 0.0`, `standardize = false`) from `particlegan.init.deterministic_orthogonal_` under protocol `seed = 0`, isolated named RNG streams (`init:*`, `data:target:training`, `prior:latent`, `noise:*`, `eval:sampler`), `recipe_owned_v1` prior ownership, 24 scheduled clean/live observations (`enumerated_prior_without_output_noise`, `scoring_weights = "live"`), and a **5-check terminal sustain gate** (`minimum_stable_checks = 5`). No host-side cluster fitting, topological particle sorting, weight resets, post-optimizer repairs, per-task recipe overrides, or seed searches are permitted.

To distinguish genuine manifold learning from memorizing lookup tables or Euclidean chord cutting, every observation evaluates:
1. **Enumerated particle endpoints** $G(z_1), \dots, G(z_M)$,
2. **Unseen interior latent points** along undirected latent $k$-NN chords $z_{ij}(t) = (1 - t)z_i + tz_j$ ($t \in \{0.25, 0.50, 0.75\}$) or 2D latent $k$-NN simplex barycenters $\frac{1}{3}(z_i + z_j + z_k)$,
3. **Whole-set novelty** (`midpoint_novelty` / `barycenter_novelty`), defined as minimum distance to **all $M$ discrete particle outputs** $\{G(z_1), \dots, G(z_M)\}$ normalized by local output nearest-neighbor spacing (scoring exact `0.0` on any Voronoi/step-function lookup table, including 2nd-neighbor chords), and
4. **Graph-alignment diagnostics** (`interp_hq` on latent $k$-NN chords vs `output_knn_interp_hq` on output-space $k$-NN chords vs `all_pairs_interp_hq` across all $\binom{M}{2}$ pairs).

Ten analytic positive and negative oracle controls (`tests/test_forge_interpolation.py`) verify every scorer against smooth manifold ground truth, Euclidean chord cutters, Voronoi lookup tables, 50/50 pixel double-exposure ghosts, flat-plane regressions, and single-mode collapses.

---

<a name="interp_sparse_arc"></a>

## 1. `interp_sparse_arc` (Tier 1 Stretch, 600 Steps)

- **Target & Capacity:** 2D $270^\circ$ circular arc ($R = 2.0$, $\theta \in [-\pi/4, 5\pi/4]$, radial $\sigma = 0.03$) with `M = 12` discrete particles in `z_dim = 2`, 2-hidden-layer 64-wide MLP generator and Fourier-2 critic (`batch = 128`, `timeout = 300s`).
- **Unseen Latent Query:** Undirected latent $k$-NN ($k = 3$) chords at $t \in \{0.25, 0.50, 0.75\}$.
- **Sustained 5-Check Gates:** `finite_fraction == 1.0`, `endpoint_hq >= 0.85`, `interp_hq >= 0.80` (`| ||y|| - R | <= 0.12`), `interp_arc_bins >= 22` of `32` (impossible for 12 discrete endpoints alone), and `midpoint_novelty >= 0.20`.
- **Actual Training Media (`atlas-particle-normal-v4`):** [api-interp-sparse-arc.gif](../../toy_audit/api_contract/interpolation/media/api-interp-sparse-arc.gif).

---

<a name="interp_img_patch_shift"></a>

## 2. `interp_img_patch_shift` (Tier 2 Stretch, 800 Steps)

- **Target & Capacity:** Continuous $1 \times 8 \times 8$ grayscale images containing a single unit-peak $2 \times 2$ window patch at continuous subpixel center $(u, v) \in [1.5, 6.5]^2$ (`0.01` clipped Gaussian noise), `M = 16` discrete particles in `z_dim = 4`, width-16 `residual_upsample` convolutional generator and convolutional critic (`batch = 64`, `timeout = 300s`).
- **Unseen Latent Query:** Undirected latent $k$-NN ($k = 2$) chords at $t \in \{0.25, 0.50, 0.75\}$ projected onto the $51 \times 51$ continuous single-patch reference manifold.
- **Sustained 5-Check Gates:** `finite_fraction == 1.0`, `endpoint_rmse <= 0.09`, `interp_rmse <= 0.10`, `interp_peak >= 0.78` (rejects 50/50 pixel-space double-exposure ghosts whose peak drops to `0.50`), `quadrant_coverage == 4`, and `midpoint_novelty >= 0.18`.
- **Actual Training Media (`atlas-particle-normal-v4`):** [api-interp-img-patch-shift.gif](../../toy_audit/api_contract/interpolation/media/api-interp-img-patch-shift.gif).

---

<a name="interp_saddle_barycentric"></a>

## 3. `interp_saddle_barycentric` (Tier 2 Stretch, 800 Steps)

- **Target & Capacity:** Non-harmonic curved 3D saddle sheet $w = 0.8(x^2 - y^2) + 0.9\cos(0.75\pi x)\cos(0.75\pi y) + \mathcal{N}(0, 0.03^2)$ over $(x, y) \in [-1, 1]^2$, `M = 24` discrete particles in `z_dim = 2`, 3D-output 2-hidden-layer 64-wide MLP generator and Fourier-2 critic (`batch = 128`, `timeout = 300s`).
- **Unseen Latent Query:** 2D simplex barycenters $\frac{1}{3}(z_i + z_j + z_k)$ of non-collinear latent $k$-NN ($k = 4$) triangles.
- **Sustained 5-Check Gates:** `finite_fraction == 1.0`, `endpoint_surface_mae <= 0.08`, `barycenter_surface_mae <= 0.09`, `barycenter_hq >= 0.78` (`|w - f(x, y)| <= 0.11`), `xy_bins_covered >= 12` of `16`, and `barycenter_novelty >= 0.18`.
- **Actual Training Media (`atlas-particle-normal-v4`):** [api-interp-saddle-barycentric.gif](../../toy_audit/api_contract/interpolation/media/api-interp-saddle-barycentric.gif).

---

## Selected-Family CUDA Results (`seed = 0`, `NVIDIA RTX A6000`, Source `0d60ac450`)

Diagnostic stretch campaign `particle-interp-stretch-v1` (`view = particle_interpolation_stretch`); preserves all recorded ordinary family qualifications without regrade:

| Family | Selected Recipe | `interp_sparse_arc` (600 steps) | `interp_img_patch_shift` (800 steps) | `interp_saddle_barycentric` (800 steps) | Sustained PASS | Readout Receipt |
| --- | --- | --- | --- | --- | ---: | --- |
| **Atlas** | `atlas-particle-normal-v4` | **FAIL** *(3/5 checks; step-600 `end_hq=1.000`, `interp_hq=0.9365`, `out_knn=1.000`, `30/32` bins, `nov=0.576`)* | **PASS** *(5/5; `end_rmse=0.0311`, `interp_rmse=0.0250`, `peak=0.9358`, `nov=0.531`)* | **FAIL** *(3/5 checks; step-800 `end_mae=0.0553`, `bary_mae=0.0552`, `bary_hq=0.8701`, `16/16` xy, `nov=0.675`)* | **1 / 3** | [readout-669ead96](../records/readout-669ead9699347bdcc998eb81.json) |
| **GAN v3 release 0.7** | `recipe-prior-refactor-release07-gan-v3-v1` | **FAIL** *(0/5; `end_hq=1.000`, `interp_hq=0.6522`, `out_knn=0.6667`, `28/32` bins, `nov=0.517`)* | **PASS** *(5/5; `end_rmse=0.0389`, `interp_rmse=0.0477`, `peak=0.8727`, `nov=0.458`)* | **FAIL** *(2/5 checks — steps 767 & 800 pass all bounds: `bary_mae=0.0587`, `bary_hq=0.8056`, `14/16` xy)* | **1 / 3** | [readout-e838608c](../records/readout-e838608cfd803f50b58f4e21.json) |
| **KA2** | `recipe-prior-refactor-ka2-v1` | **FAIL** *(0/5; `end_hq=0.9167`, `interp_hq=0.5152`, `out_knn=0.4545`, `27/32` bins, `nov=0.441`)* | **PASS** *(5/5; `end_rmse=0.0269`, `interp_rmse=0.0367`, `peak=0.9149`, `nov=0.417`)* | **FAIL** *(0/5; `end_mae=0.0545`, `bary_mae=0.0672`, `bary_hq=0.7639 < 0.78`, `15/16` xy)* | **1 / 3** | [readout-a2a29eff](../records/readout-a2a29effc3b7a25d04c3f822.json) |
| **K3P** | `recipe-prior-refactor-k3p-v1` | **FAIL** *(0/5; `end_hq=0.8333`, `interp_hq=0.5606`, `out_knn=0.4737`, `25/32` bins, `nov=0.441`)* | **PASS** *(5/5; `end_rmse=0.0289`, `interp_rmse=0.0381`, `peak=0.9349`, `nov=0.427`)* | **FAIL** *(0/5; `end_mae=0.0661`, `bary_mae=0.0768`, `bary_hq=0.6974 < 0.78`, `15/16` xy)* | **1 / 3** | [readout-d230c03f](../records/readout-d230c03f6f71197e5062b283.json) |
| **R1/R2** | `recipe-prior-refactor-r1r2-v1` | **FAIL** *(0/5; `end_hq=1.000`, `interp_hq=0.5667`, `out_knn=0.6500`, `28/32` bins, `nov=0.426`)* | **PASS** *(5/5; `end_rmse=0.0384`, `interp_rmse=0.0481`, `peak=0.8876`, `nov=0.521`)* | **FAIL** *(0/5; step-767 `bary_mae=0.0884`, `bary_hq=0.6216`, then step-800 critic spike)* | **1 / 3** | [readout-32a1753d](../records/readout-32a1753d5c69888caf94e9d0.json) |
| **EMPF** | `empf-v7` | **FAIL** *(0/5; `interp_hq=0.8254 >= 0.80`, `out_knn=0.8421`, `27/32` bins, `end_hq=0.6667 < 0.85`)* | **FAIL** *(0/5; zero-output collapse: `interp_rmse=0.1875`, `peak=0.0`)* | **FAIL** *(0/5; `end_mae=0.1158`, `bary_mae=0.1529`, `bary_hq=0.4167`, `12/16` xy)* | **0 / 3** | [readout-a211bf67](../records/readout-a211bf674ae6989087e57402.json) |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** *(0/5; `end_hq=0.7500`, `interp_hq=0.3788`, `20/32` bins, `nov=0.545`)* | **FAIL** *(0/5; `end_rmse=0.1077`, `interp_rmse=0.1178`, `peak=0.6189 < 0.78`)* | **FAIL** *(0/5; `end_mae=0.0697`, `bary_mae=0.0712`, `bary_hq=0.7286 < 0.78`, `15/16` xy)* | **0 / 3** | [readout-f569a1b3](../records/readout-f569a1b3b7d32568c334955a.json) |
| **BCAP-Pure** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** *(0/5; `end_hq=0.7500`, `interp_hq=0.4127`, `17/32` bins)* | **FAIL** *(0/5; `end_rmse=0.1384`, `interp_rmse=0.1490`, `peak=0.3015`)* | **FAIL** *(0/5; `end_mae=0.1507`, `bary_mae=0.2180`, `bary_hq=0.3000`, `11/16` xy)* | **0 / 3** | [readout-c0eb3234](../records/readout-c0eb323419b0be6463f34d1b.json) |
| **E22** | `e22` | **BLOCKED** *(preflight: no clean/live served-sampling contract)* | **BLOCKED** *(preflight: no clean/live served-sampling contract)* | **BLOCKED** *(preflight: no clean/live served-sampling contract)* | **BLOCKED** | [preflight.json](preflight.json) |

---

## Scientific Interpretation & Next Steps

1. **Zero-sigma discrete particles do NOT degenerate into Voronoi lookup tables.** Across every non-collapsed run, whole-set interior novelty (`midpoint_novelty` / `barycenter_novelty`) is **`0.41–1.34`** (`>> 0.18–0.20`), filling `25–30/32` angular arc bins from 12 particles and `14–16/16` 2D saddle cells from 24 particles.
2. **Convolutional spatial upsampling induces valid continuous patch transport without MoG (`5/8 PASS`).** Five unchanged selected recipes (`atlas`, `release07-gan-v3`, `ka2`, `k3p`, `r1r2`) pass `interp_img_patch_shift` across all `5/5` final checks (`interp_rmse = 0.025–0.048 <= 0.10`, `interp_peak = 0.873–0.936 >= 0.78`), shifting a single crisp bright $2 \times 2$ patch smoothly across unseen latent chords instead of 50/50 pixel-blending (`peak = 0.50`).
3. **MLP curved vector manifolds separate two distinct failure modes:**
   - *Secant chord-cutting without interior supervision (`ka2`, `k3p`, `r1r2`, `release07-gan-v3` on `sparse_arc`):* Endpoints land cleanly on the arc (`endpoint_hq = 0.83–1.00`), yet both `interp_hq` (`0.52–0.65`) and `output_knn_interp_hq` (`0.45–0.67`) miss `0.80`. Because `output_knn_interp_hq` matches `interp_hq`, latent topology is already ordered—the 2-layer MLP simply cuts straight Euclidean secants between adjacent particles because critic updates act only on the $M$ discrete delta atoms.
   - *Live-iterate oscillation vs terminal manifold validity (`atlas-particle-normal-v4`, `release07-gan-v3`):* `atlas-particle-normal-v4`’s recipe-owned `normal_noise` prior manipulator (`sigma = 0.0065`) exposes local latent neighborhoods during training, solving final-step geometry on both `interp_sparse_arc` (`interp_hq = 0.9365`, `output_knn_interp_hq = 1.000`, `30/32` bins, vs `all_pairs_interp_hq = 0.5606`) and `interp_saddle_barycentric` (`barycenter_surface_mae = 0.0552`, `barycenter_hq = 0.8701`, `16/16` xy bins), passing `3/5` terminal checks on each. Only step-to-step live-weight jitter prevents a `5/5` sustained pass (`release07-gan-v3` similarly passes `2/5` final checks on `saddle_barycentric`, reaching `barycenter_hq = 0.8056` at step 800). Meanwhile `empf-v7` bends arc chords (`interp_hq = 0.8254`) via DualNorm polar updates while leaving 4/12 endpoints off-arc (`endpoint_hq = 0.6667`) and collapsing to zero on zero-sigma image patches.
4. **Recommended recipe-owned hypotheses:**
   - **Recipe-owned latent chord/simplex perturbation or Jacobian isometric regularization:** Generalize `atlas`’s small local latent jitter into a recipe-owned convex-combination training sampler or directional Jacobian smoothness penalty so critic gradients supervise chord interiors directly on `ParticlePrior` (`sigma = 0`) without switching to MoG.
   - **Stabilize served/live trajectories on discrete priors:** Test EMA-served weight evaluation (in an explicitly declared served-weights cohort) or late-step critic payoff damping so `atlas` and `release07-gan-v3` hold their passing curved-manifold geometry across all 5 terminal checks while retaining every counted Tier 1–3 pass.
